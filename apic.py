import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.gpu, debug=False)

# --- Parameters ---
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
river_depth = 0.3
E = 1500  # Increased stiffness to maintain volume

flt = ti.float32

# --- Fields ---
x = ti.Vector.field(2, flt, n_particles)
v = ti.Vector.field(2, flt, n_particles)
C = ti.Matrix.field(2, 2, flt, n_particles)
J = ti.field(flt, n_particles)

grid_v = ti.Vector.field(2, flt, (n_grid, n_grid))
grid_m = ti.field(flt, (n_grid, n_grid))

gui = ti.GUI("Eisbach", res=(1024, 1024))

inflow_rate_slider = gui.slider("inflow_rate", 0.5, 5.0, step=0.1)
parabola_a_slider = gui.slider("parabola_a", 0.1, 3.0, step=0.01)
deepest_point_x_slider = gui.slider("deepest_point_x", 0.2, 1.0, step=0.01)
ground_transition_y_slider = gui.slider("ground_transition_x", 0.3, 1.0, step=0.01)

# Default slider values
inflow_rate_slider.value = 2.0
parabola_a_slider.value = 1.5
deepest_point_x_slider.value = 0.5
ground_transition_y_slider.value = 0.6

@ti.func
def riverbed(x:flt, a:flt, deepest_x:flt, ground_transition_x:flt):
    c = dx * 3
    transition_y = a * (ground_transition_x - deepest_x) ** 2 + c
    y = transition_y
    normal = tm.vec2(0, 1)
    if x < ground_transition_x:
        y = a * (x - deepest_x) ** 2 + c
        dy_dx = 2 * a * (x - deepest_x)
        normal = tm.normalize(tm.vec2(-dy_dx, 1))
    
    return y, normal

@ti.kernel
def substep(inflow: float, parabola_a: float, deepest_point_x: float, ground_transition_y: float):
    # 1. Grid Reset
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

    # 2. P2G (Particles to Grid)
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        
        # --- FIXED EOS ---
        # ti.min(..., 0) ensures we keep compression (negative values)
        # but ignore tension (positive values).
        strain = ti.min(J[p] - 1, 0) 
        
        stress = -dt * 4 * E * p_vol * strain / dx ** 2
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
        
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    # 3. Grid Operations
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * gravity
        # Boundary: Riverbed
        xi = i * dx
        y_bound, normal = riverbed(xi, parabola_a, deepest_point_x, ground_transition_y)

        # Inflow
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j] = inflow * normal
        # outflow
        if i > n_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j] = [inflow, 0.0]

        
        if (j * dx) < y_bound:
            # SLIP condition: Removed the friction ( *= 0.1 ) 
            # This prevents particles from getting "glued" to the wall in a thin line
            
            normal_component = tm.dot(grid_v[i, j], normal)
            if normal_component < 0:
                grid_v[i, j] -= normal_component * normal

        if j > n_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0

        # Velocity Clamping (Prevents explosions)
        v_max = 0.5 * dx / dt
        if tm.length(grid_v[i, j]) > v_max:
             grid_v[i, j] = tm.normalize(grid_v[i, j]) * v_max

    # 4. G2P (Grid to Particles)
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(flt, 2)
        new_C = ti.Matrix.zero(flt, 2, 2)
        
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
        
        # Respawn Logic
        if x[p].x > 1.0 - 3 * dx or x[p].x < 0 or x[p].y > 1.0:
            x[p] = [ti.random() * 2 * dx + dx,
                    ti.random() * river_depth + bound*dx,
                ]
            floor_y, normal = riverbed(x[p].x, parabola_a, deepest_point_x, ground_transition_y)
            x[p].y += floor_y
            v[p] = inflow*tm.vec2(normal.y, -normal.x)
            J[p] = 1
            C[p] = ti.Matrix.zero(flt, 2, 2)

riverbed_x = ti.field(flt, int(n_grid * 4))
riverbed_y = ti.field(flt, int(n_grid * 4))
riverbed_x.from_numpy(np.linspace(0.0, 1.0, int(n_grid * 4), dtype=np.float32))

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
        floor_y, _ = riverbed(x[p].x, parabola_a_slider.value, deepest_point_x_slider.value, ground_transition_y_slider.value)
        x[p] = [ti.random() * (1.0 - 6*dx) + 3 * dx, floor_y + ti.random() * river_depth]
        v[p] = [inflow_rate_slider.value, 0]
        J[p] = 1
        C[p] = ti.Matrix.zero(flt, 2, 2)

init()

# while gui.running and not gui.get_event(gui.ESCAPE):
#     precompute_riverbed_points(
#         parabola_a_slider.value,
#         deepest_point_x_slider.value,
#         ground_transition_y_slider.value,
#     )
#     riverbed_points = np.stack([riverbed_x.to_numpy(), riverbed_y.to_numpy()], axis=1)
    
#     # ti.profiler.print_kernel_profiler_info('trace')
#     # ti.profiler.clear_kernel_profiler_info()  # Clears all records
#     for s in range(50 * res_level):
#         substep(
#             inflow_rate_slider.value,
#             parabola_a_slider.value,
#             deepest_point_x_slider.value,
#             ground_transition_y_slider.value,
#         )
#     # ti.profiler.print_kernel_profiler_info()  # The default mode: 'count'
        
#     gui.clear(0x112F41)
#     gui.circles(riverbed_points, radius=2.0, color=0xED553B)
#     gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
#     gui.show()




particles_pos = ti.Vector.field(3, ti.f32, n_particles*10)
@ti.kernel
def set_3d():
    for p in range(n_particles):
        for d in ti.ndrange(10):
            idx = p * 10 + d
            particles_pos[idx] = tm.vec3(x[p].x, x[p].y, ti.random())

window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(5, 2, 2)

while window.running:
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

    for s in range(50 * res_level):
        substep(
            inflow_rate_slider.value,
            parabola_a_slider.value,
            deepest_point_x_slider.value,
            ground_transition_y_slider.value,
        )
    set_3d()

    scene.particles(particles_pos, color = (0.68, 0.26, 0.19), radius = 0.1)
    # Here you will get visible part from the 3rd point with (N - 4) points.
    # scene.lines(points_pos, color = (0.28, 0.68, 0.99), width = 5.0, vertex_count = N - 4, vertex_offset = 2)
    # Using indices to indicate which vertex to use
    # scene.lines(points_pos, color = (0.28, 0.68, 0.99), width = 5.0, indices = points_indices)
    # Case 1, vertex_count will be changed to N - 2 when drawing.
    # scene.lines(points_pos, color = (0.28, 0.68, 0.99), width = 5.0, vertex_count = N - 1, vertex_offset = 0)
    # Case 2, vertex_count will be changed to N - 2 when drawing.
    # scene.lines(points_pos, color = (0.28, 0.68, 0.99), width = 5.0, vertex_count = N, vertex_offset = 2)
    canvas.scene(scene)
    window.show()