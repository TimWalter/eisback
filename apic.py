# MPM-MLS in 88 lines of Taichi code, originally created by @yuanming-hu
import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.gpu, debug=False)

res_level = 1
n_particles = 8192 * 1
n_grid = tm.vec2(48 * res_level * 5, 48 * res_level)
dx = 1 / n_grid.y
domain_width = n_grid.x * dx
domain_height = n_grid.y * dx
dt = 2e-4 / res_level

p_rho = 1
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
river_depth = 0.2
E = 400

x = ti.Vector.field(2, float, n_particles)
x_normalized = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
J = ti.field(float, n_particles)

grid_v = ti.Vector.field(2, float, (n_grid.x, n_grid.y))
grid_m = ti.field(float, (n_grid.x, n_grid.y))

gui = ti.GUI("Eisbach", res=(3000, 300))
inflow_rate_slider = gui.slider("inflow_rate", 1.0, 15.0, step=0.1)
inflow_rate_slider.value = 3.5

num_points = int(domain_width*12)
riverbed_nodes_x = np.linspace(0, domain_width, num_points + 1, endpoint=True)
riverbed_nodes_y = ti.field(float, num_points + 1)
riverbed_nodes_y.from_numpy(domain_height / 2 - np.linspace(0, 0.1, num_points + 1, dtype=np.float32))

riverbed_nodes = np.stack((riverbed_nodes_x, riverbed_nodes_y.to_numpy()), axis=1)
riverbed_nodes[:, 0] /= domain_width
riverbed_nodes[:, 1] /= domain_height

@ti.func
def riverbed(x, riverbed_nodes_y):
    """Returns the y-coordinate of the riverbed at position x, and the normal vector."""
    i = int(x * num_points / domain_width)  # [0, num_points-1]
    start_node_y = riverbed_nodes_y[i]
    start_node_x = i * (domain_width / num_points)
    end_node_y = riverbed_nodes_y[i + 1]
    end_node_x = (i + 1) * (domain_width / num_points)

    tangent = ti.Vector([end_node_x - start_node_x, end_node_y - start_node_y])
    normal = ti.Vector([-tangent.y, tangent.x]).normalized()

    y = start_node_y + tangent.y / tangent.x * (x - start_node_x)

    return y, normal


@ti.kernel
def substep(inflow: float, riverbed_nodes_y: ti.template()):
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
        if i > n_grid.x - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = inflow
        # riverbed boundary
        xi = i * dx
        y_bound, normal = riverbed(xi, riverbed_nodes_y)
        y_j = int(y_bound * n_grid.y - 0.5) + 1
        normal_component = tm.dot(grid_v[i, j], normal)
        if j <= y_j and normal_component < 0:
            grid_v[i, j] -= normal_component * normal
            grid_v[i,j] *= 0.99

        if j > n_grid.y - bound and grid_v[i, j].y > 0:
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

        # if the particle goes out of bounds, respawn it at the left side
        y, normal = riverbed(x[p].x, riverbed_nodes_y)
        if x[p].x > domain_width - 3 * dx or x[p].x < dx or x[p].y < y or x[p].y > domain_height - 3 * dx:
            x[p] = [ti.random() * 3 * dx + dx, ti.random() * river_depth]
            x[p] += [0.0, y]
            v[p] = [normal.y, -normal.x]
            v[p] *= inflow
            J[p] = 1
            C[p] = ti.Matrix.zero(float, 2, 2)


@ti.kernel
def init(riverbed_nodes_y: ti.template()):
    for p in range(n_particles):
        x[p] = [ti.random() * 0.95 * domain_width + 3 * dx, ti.random() * river_depth]
        y, normal = riverbed(x[p].x, riverbed_nodes_y)
        x[p] += [0.0, y]
        v[p] = [normal.y * 2, -normal.x]
        J[p] = 1
        C[p] = ti.Matrix.zero(float, 2, 2)


init(riverbed_nodes_y)
save = gui.button('Save')

while gui.running and not gui.get_event(gui.ESCAPE):
    for s in range(200 * res_level):
        substep(inflow_rate_slider.value, riverbed_nodes_y)

    # Render riverbed nodes
    gui.clear(0x112F41)
    gui.circles(riverbed_nodes, radius=5.0, color=0xED553B)  # Draw riverbed
    gui.lines(riverbed_nodes[:-1], riverbed_nodes[1:], radius=1.0, color=0xED553B)

    particle_vis = x.to_numpy()
    particle_vis[:, 0] /= domain_width
    particle_vis[:, 1] /= domain_height
    gui.circles(particle_vis, radius=2.5, color=0x068587)

    if gui.is_pressed(ti.GUI.LMB):
        mouse_x, mouse_y = gui.get_cursor_pos()
        idx = int(mouse_x * num_points + 0.5)
        riverbed_nodes_y[idx] = mouse_y * domain_height
        riverbed_nodes[idx, 1] = riverbed_nodes_y[idx] / domain_height

    if gui.is_pressed(save):
        print(riverbed_nodes_y)

    gui.show()
