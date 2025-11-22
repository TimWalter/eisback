import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.gpu, debug=True)

n_particles = 8192
n_grid = tm.vec2(512, 512)
dp = 1 / n_grid
dt = 2e-4

p_rho = 1
cell_area = dp.x * dp.y
p_vol = 0.25 * cell_area
p_mass = p_vol * p_rho
gravity = 9.8

source_thickness = dp.x
river_depth = 0.1
# Ground properties
parabola_a = 0.5
deepest_point_x = 0.4
ground_transition_y = 0.1

# X-point at which the riverbed transitions to flat
transition_to_flat = 1 / 2

E = 400

# Positions
x = ti.Vector.field(2, float, n_particles)
# Velocities
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
J = ti.field(float, n_particles)

grid_v = ti.Vector.field(2, float, (n_grid.x + 2, n_grid.y + 2))
grid_m = ti.field(float, (n_grid.x + 2, n_grid.y + 2))


@ti.func
def riverbed(x):
    """Returns the y-coordinate of the riverbed at position x, and the normal vector."""
    # Parabolic riverbed profile
    y = parabola_a * (x - deepest_point_x) ** 2
    y = ti.min(y, ground_transition_y)+ 3*dp.y
    # Normal vector calculation
    dy_dx = 2 * parabola_a * (x - deepest_point_x)
    normal = tm.vec2(-dy_dx, 1)

    normal /= tm.length(normal)

    return y, normal


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        # Periodic wrapping in x-direction
        if x[p].x <= 0.0 or x[p].x >= (1.0 - dp.x):
            x[p] = [ti.random() * source_thickness, ti.random() * river_depth]
            x[p] += [0.0, 0.5 + riverbed(x[p].x)[0]]
            v[p] = [0.5, 0]
            J[p] = 1
            C[p] = ti.Matrix.zero(float, 2, 2)

        y_bound_lower, normal = riverbed(x[p].x)
        if x[p].y < y_bound_lower:
            x[p].y = y_bound_lower
            v[p] = [v[p].x, 0.0]
            # v_normal = v[p].dot(normal) * normal
            # v_tangent = v[p] - v_normal
            # v[p] = v_tangent - 0.5 * v_normal  # simple restitution
            J[p] = 1
            C[p] = ti.Matrix.zero(float, 2, 2)
        if x[p].y >= (1.0 - dp.y):
            x[p].y = 1.0 - dp.y
            v[p].y = tm.min(0, v[p].y)
            J[p] = 1
            C[p] = ti.Matrix.zero(float, 2, 2)

        Xp = x[p] / dp
        base = int(Xp - 0.5)  # round (Xp - 1), such that the kernel centred at 0
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / cell_area
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dp
            weight = w[i].x * w[j].y
            grid_v[base + 1 + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + 1 + offset] += weight * p_mass

    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * gravity
    for p in x:
        Xp = x[p] / dp
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dp
            weight = w[i].x * w[j].y
            g_v = grid_v[base + 1 + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / cell_area
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C


@ti.kernel
def init():
    for p in range(n_particles):
        x[p] = [ti.random(), ti.random() * river_depth]
        x[p] += [0.0, riverbed(x[p].x)[0]]
        v[p] = [0.5, 0]
        J[p] = 1
        C[p] = ti.Matrix.zero(float, 2, 2)


riverbed_x = ti.field(float, int(n_grid.x * 4))
riverbed_x.from_numpy(np.linspace(0.0, 1.0, int(n_grid.x * 4), dtype=np.float32))
riverbed_y = ti.field(float, int(n_grid.x * 4))


@ti.kernel
def precompute_riverbed_points():
    for i in range(int(n_grid.x * 4)):
        rx = riverbed_x[i]
        riverbed_y[i] = riverbed(rx)[0]



init()
precompute_riverbed_points()
riverbed_points = np.stack([riverbed_x.to_numpy(), riverbed_y.to_numpy()], axis=1)

# Output fields for normalized_m and v, only for plotting
grid_m_normalized = ti.field(ti.u8, (int(n_grid.x) + 2, int(n_grid.y) + 2))
grid_v_normalized = ti.Vector.field(2, ti.u8, (int(n_grid.x) + 2, int(n_grid.y) + 2))

@ti.kernel
def normalized_m_scalar():
    max_scalar = 0.0
    
    # First pass: find max
    for i, j in grid_m:
        max_scalar = ti.atomic_max(max_scalar, grid_m[i,j])

    # Second pass: normalize
    for i, j in grid_m:
        grid_m_normalized[i,j] = ti.cast(grid_m[i,j] / max_scalar, ti.u8)


@ti.kernel
def normalized_v_vector():
    max_scalar = 0.0
    
    # First pass: find max magnitude
    for i, j in grid_v:
        max_scalar = ti.atomic_max(max_scalar, tm.length(grid_v[i,j]))

    # Second pass: normalize
    for i, j in grid_v:
        grid_v_normalized[i,j] = ti.cast(ti.abs(grid_v[i,j]) / max_scalar, ti.u8)


gui = ti.GUI("MPM88")
gui2 = ti.GUI("Mass", res=(int(n_grid.x) + 2, int(n_grid.y) + 2), fast_gui=False)
gui3 = ti.GUI("Velocity", res=(int(n_grid.x) + 2, int(n_grid.y) + 2), fast_gui=False)
toggle = True
while gui.running and not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    if gui.is_pressed(ti.GUI.SPACE) and toggle:
        toggle = False
        for s in range(10):
            substep()
        print("total_mass", np.sum(grid_m.to_numpy()))
    if not gui.is_pressed(ti.GUI.SPACE):
        toggle = True

    gui.clear(0x000000)
    gui2.clear(0x000000)
    gui3.clear(0x000000)
    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    gui.circles(riverbed_points, radius=2.0, color=0xED553B)  # Draw riverbed


    normalized_m_scalar()
    normalized_v_vector()
    gui2.set_image(grid_m_normalized)
    gui3.set_image(grid_v_normalized)

    gui.show()
    gui2.show()
    gui3.show()