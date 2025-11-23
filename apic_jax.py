import jax
import jax.numpy as jnp
import numpy as np
import taichi as ti
from functools import partial

ti.init(arch=ti.gpu)

# --- Configuration ---
res_level = 2
n_particles = 8192
n_grid = 128 * res_level
dx = 1.0 / n_grid
dt = 1e-4 / res_level

p_rho = 1.0
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
river_depth = 0.1
E = 1500.0 

# --- State Initialization ---
key = jax.random.PRNGKey(0)

def init_state(key):
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.uniform(k1, (n_particles, 2)) * 0.4
    x = x.at[:, 1].add(0.1)
    
    v = jnp.zeros((n_particles, 2))
    C = jnp.zeros((n_particles, 2, 2))
    J = jnp.ones((n_particles,))
    
    return {"x": x, "v": v, "C": C, "J": J}

# --- Physics Functions ---

def get_riverbed(x_coord, a, deepest_x, ground_transition_x):
    c = dx * 3
    
    # Logic for parabolic section
    y_para = a * (x_coord - deepest_x) ** 2 + c
    dy_dx = 2 * a * (x_coord - deepest_x)
    
    norm_x_para = -dy_dx
    norm_y_para = jnp.ones_like(x_coord)
    norm_len = jnp.sqrt(norm_x_para**2 + norm_y_para**2)
    nx_para = norm_x_para / norm_len
    ny_para = norm_y_para / norm_len

    # Logic for flat section
    y_flat = a * (ground_transition_x - deepest_x) ** 2 + c
    nx_flat = jnp.zeros_like(x_coord)
    ny_flat = jnp.ones_like(x_coord)

    mask = x_coord < ground_transition_x
    
    y = jnp.where(mask, y_para, y_flat)
    nx = jnp.where(mask, nx_para, nx_flat)
    ny = jnp.where(mask, ny_para, ny_flat)
    
    return y, jnp.stack([nx, ny], axis=-1)

def substep(state, params):
    """
    Single simulation step. 
    Note: No @jax.jit here, because this is now the body of the scan.
    """
    x, v, C, J = state["x"], state["v"], state["C"], state["J"]
    
    # --- 1. Particles to Grid (P2G) ---
    base = (x / dx - 0.5).astype(int)
    fx = x / dx - base
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    
    strain = jnp.minimum(J - 1, 0)
    stress = -dt * 4 * E * p_vol * strain / dx ** 2
    
    stress_mat = jnp.zeros((n_particles, 2, 2))
    stress_mat = stress_mat.at[:, 0, 0].set(stress)
    stress_mat = stress_mat.at[:, 1, 1].set(stress)
    affine = stress_mat + p_mass * C

    grid_v_temp = jnp.zeros((n_grid, n_grid, 2))
    grid_m_temp = jnp.zeros((n_grid, n_grid))

    for i in range(3):
        for j in range(3):
            offset = jnp.array([i, j])
            dpos = (offset - fx) * dx
            weight = w[i][:, 0] * w[j][:, 1]
            
            momentum_contrib = weight[:, None] * (p_mass * v + jnp.einsum('nij,nj->ni', affine, dpos))
            mass_contrib = weight * p_mass
            
            idx_x = jnp.clip(base[:, 0] + i, 0, n_grid - 1)
            idx_y = jnp.clip(base[:, 1] + j, 0, n_grid - 1)
            
            grid_v_temp = grid_v_temp.at[idx_x, idx_y, :].add(momentum_contrib)
            grid_m_temp = grid_m_temp.at[idx_x, idx_y].add(mass_contrib)

    # --- 2. Grid Operations ---
    grid_mask = grid_m_temp > 1e-10
    grid_v_temp = jnp.where(grid_mask[:, :, None], grid_v_temp / grid_m_temp[:, :, None], grid_v_temp)
    grid_v_temp = grid_v_temp.at[:, :, 1].add(-dt * gravity)

    # Boundaries
    grid_idx_x, grid_idx_y = jnp.meshgrid(jnp.arange(n_grid), jnp.arange(n_grid), indexing='ij')
    grid_pos_x = grid_idx_x * dx
    grid_pos_y = grid_idx_y * dx
    
    # 2a. Riverbed
    bed_y, bed_n = get_riverbed(grid_pos_x, params['a'], params['deep_x'], params['trans_y'])
    is_below_bed = grid_pos_y < bed_y
    
    v_dot_n = jnp.sum(grid_v_temp * bed_n, axis=-1)
    correction = (v_dot_n < 0) * v_dot_n
    grid_v_corrected = grid_v_temp - correction[:, :, None] * bed_n
    grid_v_temp = jnp.where(is_below_bed[:, :, None], grid_v_corrected, grid_v_temp)

    # 2b. Box Boundaries (FIXED CONCATENATION ERROR HERE)
    
    # Inflow (Left)
    mask_inflow = (grid_idx_x < bound) & (grid_v_temp[:, :, 0] < 0)
    grid_v_temp = jnp.where(mask_inflow[:, :, None], jnp.array([params['inflow'], 0.0]), grid_v_temp)
    
    # Outflow (Right)
    mask_outflow = (grid_idx_x > n_grid - bound) & (grid_v_temp[:, :, 0] > 0)
    outflow_target = jnp.stack([jnp.full_like(grid_v_temp[..., 0], params['inflow']), grid_v_temp[..., 1]], axis=-1)
    grid_v_temp = jnp.where(mask_outflow[:, :, None], outflow_target, grid_v_temp)
    
    # Ceiling
    mask_ceil = (grid_idx_y > n_grid - bound) & (grid_v_temp[:, :, 1] > 0)
    ceil_target = jnp.stack([grid_v_temp[..., 0], jnp.zeros_like(grid_v_temp[..., 1])], axis=-1)
    grid_v_temp = jnp.where(mask_ceil[:, :, None], ceil_target, grid_v_temp)

    # 2c. Velocity Clamping
    v_max = 0.5 * dx / dt
    v_len = jnp.linalg.norm(grid_v_temp, axis=-1, keepdims=True)
    grid_v_temp = jnp.where(v_len > v_max, grid_v_temp / v_len * v_max, grid_v_temp)

    # --- 3. Grid to Particles (G2P) ---
    new_v = jnp.zeros_like(v)
    new_C = jnp.zeros_like(C)
    
    for i in range(3):
        for j in range(3):
            offset = jnp.array([i, j])
            dpos = (offset - fx) * dx
            weight = w[i][:, 0] * w[j][:, 1]
            
            idx_x = jnp.clip(base[:, 0] + i, 0, n_grid - 1)
            idx_y = jnp.clip(base[:, 1] + j, 0, n_grid - 1)
            g_v = grid_v_temp[idx_x, idx_y]
            
            new_v = new_v + weight[:, None] * g_v
            outer = jnp.einsum('ni,nj->nij', g_v, dpos)
            new_C = new_C + 4 * weight[:, None, None] * outer / dx ** 2

    x = x + dt * new_v
    J = J * (1 + dt * jnp.trace(new_C, axis1=1, axis2=2))
    J = jnp.clip(J, 0.6, 1.4)
    
    # Respawn Logic (Deterministic)
    is_out = (x[:, 0] > 1.0 - 3*dx) | (x[:, 0] < 0) | (x[:, 1] > 1.0)
    
    spawn_x = jnp.mod(x[:, 0] * 7919, 2 * dx) + dx 
    spawn_y_base, _ = get_riverbed(spawn_x, params['a'], params['deep_x'], params['trans_y'])
    spawn_y = spawn_y_base + jnp.mod(x[:, 1] * 691, river_depth)
    
    x_reset = jnp.stack([spawn_x, spawn_y], axis=-1)
    v_reset = jnp.array([params['inflow'], 0.0])
    
    x = jnp.where(is_out[:, None], x_reset, x)
    v = jnp.where(is_out[:, None], v_reset, new_v)
    J = jnp.where(is_out, 1.0, J)
    C = jnp.where(is_out[:, None, None], jnp.zeros((2, 2)), new_C)
    
    return {"x": x, "v": v, "C": C, "J": J}

@partial(jax.jit, static_argnums=(2,))
def update_frame(state, params, num_substeps):
    """
    Updates the state for 'num_substeps' iterations using jax.lax.scan.
    This fuses the entire loop into one kernel.
    """
    
    # scan requires function signature (carry, x) -> (new_carry, y)
    # We use a lambda to bind 'params' which are constant across the frame
    step_fn = lambda s, _: (substep(s, params), None)
    
    # We don't need the per-step output, so we ignore the second return value
    final_state, _ = jax.lax.scan(step_fn, state, None, length=num_substeps)
    
    return final_state

# --- Main Execution Loop ---

state = init_state(key)

gui = ti.GUI("JAX Eisbach (Scan)", res=(512, 512))
inflow_rate_slider = gui.slider("inflow_rate", 0.5, 5.0, step=0.1)
parabola_a_slider = gui.slider("parabola_a", 0.1, 3.0, step=0.01)
deepest_point_x_slider = gui.slider("deepest_point_x", 0.2, 1.0, step=0.01)
ground_transition_y_slider = gui.slider("ground_transition_x", 0.3, 1.0, step=0.01)

# Defaults
inflow_rate_slider.value = 2.0
parabola_a_slider.value = 1.5
deepest_point_x_slider.value = 0.5
ground_transition_y_slider.value = 0.6

substeps_per_frame = 100 * res_level

# Warmup compile
print("Compiling JAX kernel with scan...")
dummy_params = {'inflow': 2.0, 'a': 1.5, 'deep_x': 0.5, 'trans_y': 0.6}
state = update_frame(state, dummy_params, substeps_per_frame)
print("Compilation complete.")

while gui.running:
    params = {
        'inflow': inflow_rate_slider.value,
        'a': parabola_a_slider.value,
        'deep_x': deepest_point_x_slider.value,
        'trans_y': ground_transition_y_slider.value
    }
    
    # This is now a single JAX call
    state = update_frame(state, params, substeps_per_frame)
    
    # Retrieve data for rendering
    # Block_until_ready ensures we don't queue frames faster than GPU can compute
    np_x = np.array(state["x"].block_until_ready())
    
    gui.clear(0x112F41)
    
    bed_x_vis = np.linspace(0, 1, 512)
    bed_y_vis, _ = get_riverbed(jnp.array(bed_x_vis), params['a'], params['deep_x'], params['trans_y'])
    bed_points = np.stack([bed_x_vis, np.array(bed_y_vis)], axis=1)
    gui.circles(bed_points, radius=2.0, color=0xED553B)
    
    gui.circles(np_x, radius=1.5, color=0x068587)
    
    gui.show()