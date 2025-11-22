import taichi as ti

ti.init(arch=ti.gpu, debug=True)


# system params
n_particles = 8192
domain_x = 1
domain.y = 1
gravity = 1

# floor
parabola_a = 0.5
parabola_c = 0.1
deepest_point_x = 0.4
ground_transition_y = 0.1 + parabola_c

# water params
viscosity = 1
interaction_radius = 0.1
interaction_radius_sq = interaction_radius ** 2
mass = 1 

# kernel
h = interaction_radius
h_sq = h * h
POLY6_2D_CONST = 4.0 / (math.pi * h**8)  # Poly6 (Density): 4 / (pi * h^8)
SPIKY_GRAD_2D_CONST = -30.0 / (math.pi * h**5)  # Spiky (Gradient): -30 / (pi * h^5)

# particle data
x = ti.Vector.field(2, float, n_particles) # positions
v = ti.Vector.field(2, float, n_particles) # velocities
f = ti.Vector.field(2, float, n_particles) # forces
p = ti.field(float, n_particles) # pressure
d = ti.field(float, n_particles) # density




def riverbed(x: np.ndarray) -> tuple[float | np.ndarray, tm.vec2]:
    """Returns the y-coordinate of the riverbed at position x, and the normal vector."""
    # Parabolic riverbed profile
    y = parabola_a * (x - deepest_point_x) ** 2 + parabola_c
    y = np.minimum(y, ground_transition_y)
    # Normal vector calculation
    dy_dx = 2 * parabola_a * (x - deepest_point_x)
    normal = tm.vec2(-dy_dx, 1).normalized()

    return y, normal

@ti.func
def tait_pressure(rho, rho0, c0, gamma):
    B = c0 * c0 * rho0 / gamma
    return B * ((rho / rho0)**gamma - 1.0)

@ti.func
def poly6_value(r_sq, h_sq):
    result = 0.0
    if r_sq < h_sq:
        # Formula: (h^2 - r^2)^3
        diff = h_sq - r_sq
        result = POLY6_2D_CONST * diff * diff * diff
    return result

@ti.func
def spiky_gradient(r, r_len, h):
    res = ti.Vector([0.0, 0.0])
    if 0 < r_len < h:
        # Formula: (h - r)^2
        diff = h - r_len
        
        # The raw scalar value of the gradient magnitude
        grad_mag = SPIKY_GRAD_2D_CONST * diff * diff
        
        # Multiply by direction (r / r_len) to get the vector gradient
        # We perform the division here to normalize the direction
        res = r * (grad_mag / r_len)
    return res

@ti.kernel
def substep():

    # 1. pass: density calcualtion
    for i in x:
        d[i] = 0.0
        for j in x:
            r = x[i] - x[j]
            r_sq = r.norm_sqr()
            if r_sq < h_sq:
                d[i] += mass * poly6_value(r_sq, h_sq)

        # avoid division by 0 if density is too low (recommended by gemini)
        d[i] = ti.max(d[i], mass * POLY6_2D_CONST * h_sq**3)

    # 2. pass: force calculation
    for i in x:
        f[i] = ti.Vector([0.0, -gravity])
        
        # simplified tait equation
        stiffness = 500.0 
        rest_density = 1.0 # Assuming 1 for simplicity
        p[i] = stiffness * (d[i] - rest_density)


# 2. Compute Forces (The Pairwise Interaction)
for i in particles:
    i.force = gravity
    # Convert density to pressure (Equation of State)
    i.pressure = k * (i.density - rest_density) 

    for 
    
    for j in neighbors_of(i):
        # Calculate Pressure Force Gradient
        # Note: We use Gradient of Kernel (GradKernel) here
        pressure_force = -j.mass * (i.pressure/i.density**2 + j.pressure/j.density**2) * GradKernel(dist, h)
        
        # Calculate Viscosity (friction between particles)
        viscosity_force = viscosity_constant * (j.vel - i.vel) * ...
        
        i.force += pressure_force + viscosity_force

# 3. Integration (Move particles)
for i in particles:
    i.vel += i.force / i.mass * dt
    i.pos += i.vel * dt