# MPM-MLS in 88 lines of Taichi code, originally created by @yuanming-hu
import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.gpu, debug=False)

res_level = 2
n_particles = 8192
n_grid = 128 * res_level
dx = 1 / n_grid
dt = 2e-4 / res_level

p_rho = 1
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
gravity = 0.98
bound = 3
river_depth = 0.1
E = 400

x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
J = ti.field(float, n_particles)

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))

gui = ti.GUI("Eisbach")

inflow_rate_slider = gui.slider("inflow_rate", 0.5, 5.0, step=0.1)
parabola_a_slider = gui.slider("parabola_a", 0.1, 3.0, step=0.01)
deepest_point_x_slider = gui.slider("deepest_point_x", 0.2, 1.0, step=0.01)
ground_transition_y_slider = gui.slider("ground_transition_x", 0.3, 1.0, step=0.01)


@ti.func
def riverbed(x, a, deepest_x, ground_transition_x):
    """Returns the y-coordinate of the riverbed at position x, and the normal vector."""
    # Parabolic riverbed profile
    c = dx * 3
    transition_y = a * (ground_transition_x - deepest_x) ** 2 + c
    y = transition_y

    normal = tm.vec2(0, 1)
    if x < ground_transition_x:
        y = a * (x - deepest_x) ** 2 + c
        # Normal vector calculation
        dy_dx = 2 * a * (x - deepest_x)
        normal = tm.vec2(-dy_dx, 1)

        normal /= tm.length(normal)
    else:
        y = transition_y
        normal = tm.vec2(0, 1)
    return y, normal


@ti.kernel
def substep(inflow:float, parabola_a: float, deepest_point_x: float, ground_transition_y: float):
    # for _ in ti.static(ti.ndrange(50)):
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx ** 2
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * gravity

        # Inflow
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = inflow
        # outflow
        if i > n_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = inflow

        # riverbed boundary
        xi = i * dx
        y_bound, normal = riverbed(xi, parabola_a, deepest_point_x, ground_transition_y)
        y_j = int(y_bound * n_grid - 0.5) + 1
        normal_component = tm.dot(grid_v[i, j], normal)
        if j <= y_j and normal_component < 0:
            grid_v[i, j] -= normal_component * normal

        if j > n_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx ** 2
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C

        if x[p].x > 1.0 - 3 * dx:
            x[p] = [ti.random() * 2 * dx + dx, ti.random() * river_depth]
            y, normal = riverbed(
                x[p].x, parabola_a, deepest_point_x, ground_transition_y
            )
            x[p] += [0.0, y]
            v[p] = inflow*tm.vec2(normal.y, -normal.x)
            J[p] = 1
            C[p] = ti.Matrix.zero(float, 2, 2)


riverbed_x = ti.field(float, int(n_grid * 4))
riverbed_x.from_numpy(np.linspace(0.0, 1.0, int(n_grid * 4), dtype=np.float32))
riverbed_y = ti.field(float, int(n_grid * 4))


@ti.kernel
def precompute_riverbed_points(parabola_a: float, deepest_point_x: float, ground_transition_y: float):
    for i in range(int(n_grid * 4)):
        rx = riverbed_x[i]
        riverbed_y[i], _ = riverbed(
            rx,
            parabola_a,
            deepest_point_x,
            ground_transition_y,
        )


@ti.kernel
def init():
    for p in range(n_particles):
        x[p] = [ti.random() * (1.0 - 6*dx) + 3 * dx, ti.random() * river_depth]
        x[p] += [
            0.0,
            riverbed(
                x[p].x,
                parabola_a_slider.value,
                deepest_point_x_slider.value,
                ground_transition_y_slider.value,
            )[0],
        ]
        v[p] = [1.0, 0]
        J[p] = 1
        C[p] = ti.Matrix.zero(float, 2, 2)


init()
while gui.running and not gui.get_event(gui.ESCAPE):
    precompute_riverbed_points(
            parabola_a_slider.value,
            deepest_point_x_slider.value,
            ground_transition_y_slider.value,)
    riverbed_points = np.stack([riverbed_x.to_numpy(), riverbed_y.to_numpy()], axis=1)
    for s in range(50*res_level):
        substep(
            inflow_rate_slider.value,
            parabola_a_slider.value,
            deepest_point_x_slider.value,
            ground_transition_y_slider.value,
        )
    gui.clear(0x112F41)
    gui.circles(riverbed_points, radius=2.0, color=0xED553B)  # Draw riverbed
    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    gui.show()
