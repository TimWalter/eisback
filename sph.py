import taichi as ti
import math
import numpy as np
import taichi.math as tm


ti.init(arch=ti.gpu)



# system params
particles_x = 100
particles_y = 150
n_particles = particles_x * particles_y
domain_x = 10
domain_y = 10
gravity = 9.81 * 5
dt = 1e-4


# floor
parabola_a = 2
parabola_c = 0.1
deepest_point_x = 4
ground_transition_y = 1 


# water params
particle_radius = 0.01
particle_diameter = 2 * particle_radius
dx = particle_diameter
viscosity_mu = 0.005

rho0 = 1000 # kg/m^3
c0 = 80 # [m/s] speed of sound in (our) water 
interaction_radius = dx * 2.5
interaction_radius_sq = interaction_radius ** 2
mass = dx * dx * rho0

# kernel
h = interaction_radius
h_sq = h * h
POLY6_2D_CONST = 4.0 / (math.pi * h**8)  # Poly6 (Density): 4 / (pi * h^8)
SPIKY_GRAD_2D_CONST = -30.0 / (math.pi * h**5)  # Spiky (Gradient): -30 / (pi * h^5)

# tait pressure
gamma = 7 
B = c0*c0*rho0/gamma


# particle data
x = ti.Vector.field(2, float, n_particles) # positions
v = ti.Vector.field(2, float, n_particles) # velocities
f = ti.Vector.field(2, float, n_particles) # forces
f_old = ti.Vector.field(2, float, n_particles) # old_forces
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


# standard density estimation kernel 
@ti.func
def poly6_value(r_sq, h_sq):
    result = 0.0
    if r_sq < h_sq:
        # Formula: (h^2 - r^2)^3
        diff = h_sq - r_sq
        result = POLY6_2D_CONST * diff * diff * diff
    return result

# pressure force gradient for pair wise force calculations
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


# convert desnity to pressure 
@ti.func
def tait_pressure(rho, rho0, c0, gamma, B):
    return B * ((rho / rho0)**gamma - 1.0)


@ti.func
def viscosity_laplacian(r, h):
    # Müller 2003 Viscosity Kernel Laplacian
    # ∇²W(r) = (45 / (π * h^6)) * (h - r)
    res = 0.0
    if r < h:
        res = (45.0 / (math.pi * h**6)) * (h - r)
    return res


@ti.kernel
def init():
    for i in range(particles_x):
        for j in range(particles_y):

            x[i * particles_y + j] = [i * dx, j * dx]
            v[i * particles_y + j] = [0.01 * (ti.random() - 0.5 + 5), 0.01 * (ti.random() - 0.5) + 1]


    # for p in range(n_particles):
    #     x[p] = [ti.random() * domain_x / 5, ti.random() * domain_y / 2 ] 
        # x[p] += [0.0, riverbed(x[p].x)[0]]
        # v[p] = [1.0, 0]
        # J[p] = 1
        # C[p] = ti.Matrix.zero(float, 2, 2)


@ti.kernel
def force_update():
    # 0. pass store old forces
    for i in range(n_particles):
        f_old[i] = f[i]      # copy force
        f[i] = ti.Vector([0.0, 0.0])   # reset force
   
    # 1. pass: density calcualtion
    for i in range(n_particles):
        d[i] = 0.0
        for j in range(n_particles):
            r = x[i] - x[j]
            r_sq = r.norm_sqr()
            if r_sq < h_sq:
                d[i] += mass * poly6_value(r_sq, h_sq)

        # avoid division by 0 if density is too low (recommended by gemini)
        d[i] = ti.max(d[i], mass * POLY6_2D_CONST * h_sq**3)

    # 2. pass: pressure compute
    for i in x:
        p[i] = tait_pressure(d[i], rho0, c0, gamma, B)

    # 3. pass: force calculation
    for i in range(n_particles):
        f[i] = ti.Vector([0.0, -gravity])

        for j in range(n_particles):
            r = x[i] - x[j]
            r_len = r.norm()

            if 0 < r_len < interaction_radius:
                # SPH Force Formula: F = -mi * mj * (Pi/rhoi^2 + Pj/rhoj^2) * GradW

                term_i = p[i] / (d[i] * d[i])
                term_j = p[j] / (d[j] * d[j])

                f_pressure = -mass * mass * (term_i + term_j) * spiky_gradient(r, r_len, h)
                f[i] += f_pressure

                # Viscosity Force
                v_rel = v[j] - v[i]
                f_viscosity = viscosity_mu * mass * (v_rel / d[j]) * viscosity_laplacian(r_len, h)
                f[i] += f_viscosity


@ti.kernel
def apply_boundary_conditions():
    for i in range(n_particles):
        if x[i][0] < 0: 
            x[i][0] = domain_x
            # v[i][0] = - v[i][0]
            # v[i][1] *=0.9 
        if x[i][0] > domain_x:
            x[i][0] = 0
            # v[i][0] = - v[i][0]
            # v[i][1] *=0.9 
        if x[i][1] < 0:
            x[i][1] = 0
            v[i][1] = -v[i][1]
            # v[i][0] *=0.9 


@ti.kernel
def verlet_pass_1():
    # 1. Update Position & Apply Boundaries
    for i in range(n_particles): 
        acc = f[i] / mass
        x[i] += v[i] * dt + 0.5 * acc * (dt * dt)
        

@ti.kernel
def verlet_pass_2():
    # 2. Update Velocity using Average Acceleration
    for i in range(n_particles): 
        acc_avg = (f[i] + f_old[i]) / mass
        v[i] += 0.5 * acc_avg * dt


def velocity_verlet_step(): 
    verlet_pass_1()
    
    apply_boundary_conditions()
    force_update()
    
    verlet_pass_2()



gui = ti.GUI("MPM88", res=(800,700))
init()

while gui.running and not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):

    for s in range(50):
        velocity_verlet_step()
    gui.clear(0x000000)

    pos_np = x.to_numpy()
    pos_np[:, 0] /= domain_x
    pos_np[:, 1] /= domain_y

    gui.circles(pos_np, radius=2, color=0x068587)
    # gui.circles(riverbed_points, radius=2.0, color=0xED553B)  # Draw riverbed
    # gui.circles(x.to_numpy(), radius=1.5, color=0x068587)

    gui.show()
