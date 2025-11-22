# MPM-MLS in 88 lines of Taichi code, originally created by @yuanming-hu
import taichi as ti
import taichi.math as tm
import numpy as np
import jaxtyping as jt

ti.init(arch=ti.gpu, debug=True)

n_particles = 8192
n_grid = 128
dx = 1 / n_grid
dt = 2e-4

p_rho = 1
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
gravity = 9.8

river_length = 0.9
river_depth = 0.2
# Ground properties
parabola_a = 0.5
parabola_c = 0.1
deepest_point_x = 0.4
ground_transition_y = 0.1 + parabola_c


# X-point at which the riverbed transitions to flat
transition_to_flat = river_length / 2

y_bound_lower = 0
y_bound_upper = 5

bound = 3
E = 400

# Positions
x = ti.Vector.field(2, float, n_particles)
# Velocities
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
J = ti.field(float, n_particles)

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))


def riverbed(x: np.ndarray) -> tuple[float | np.ndarray, tm.vec2]:
    """Returns the y-coordinate of the riverbed at position x, and the normal vector."""
    # Parabolic riverbed profile
    y = parabola_a * (x - deepest_point_x) ** 2 + parabola_c
    y = np.minimum(y, ground_transition_y)
    # Normal vector calculation
    dy_dx = 2 * parabola_a * (x - deepest_point_x)
    normal = tm.vec2(-dy_dx, 1).normalized()

    return y, normal


@ti.kernel
def substep():
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
        # if i < bound and grid_v[i, j].x < 0:
        #     grid_v[i, j].x = 0
        # if i > n_grid - bound and grid_v[i, j].x > 0:
        #     grid_v[i, j].x = 0
        # if j < bound and grid_v[i, j].y < 0:
        #     grid_v[i, j].y = 0
        # if j > n_grid - bound and grid_v[i, j].y > 0:
        #     grid_v[i, j].y = 0

    
    
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


@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [ti.random() * river_length, ti.random() * river_depth]
        # Ensure particles start withing [0, 1] x- and y-range with a small margin
        x[i] = ti.math.clamp(x[i], 2 * dx, 1.0 - 2 * dx)
        v[i] = [0.8, 0]
        J[i] = 1


init()
gui = ti.GUI("MPM88")
while gui.running and not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    for s in range(50):
        substep()
    gui.clear(0x000000)
    gui.circles(x.to_numpy(), radius=2.5, color=0xED553B)

    # Draw riverbed
    riverbed_x = np.linspace(0.0, 1.0, n_grid*4)
    riverbed_y = riverbed(riverbed_x)[0]
    riverbed_points = np.stack([riverbed_x, riverbed_y], axis=1)
    gui.circles(riverbed_points, radius=2.0, color=0x068587)  # Draw riverbed

    gui.show()