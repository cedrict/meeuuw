import numpy as np
from constants import *

###################################################################################################

Lx = 3000e3
Lz = 750e3
eta_ref = 1e21
p_scale = 1e6
p_unit = "MPa"
vel_scale = cm / year
vel_unit = "cm/yr"
time_scale = year
time_unit = "yr"
end_time = 50e6 * year
every_solution = 1
every_solution_png = 10
every_swarm_vtu = 1
every_swarm_png = 10
every_swarm_ascii = 10
averaging = "arithmetic"
debug_ascii = False
nparticle_per_dim = 7
particle_distribution = 0  # 0: random, 1: reg
# remove_rho_profile=True

nmat=3

nelz = 60
nelx = int(Lx / Lz * nelz * 1.25)
nstep = 1

###################################################################################################

def particle_layout(nparticle, nmat, swarm_x, swarm_z, swarm_rad, swarm_theta, Lx, Lz):

    swarm_wf = np.zeros((nmat, nparticle), dtype=np.float64)

    for ip in range(0, nparticle):
        swarm_wf[1, ip] = 1 # mantle

        # sticky air
        if swarm_z[ip] > Lz - 50e3:
            swarm_wf[0, ip] = 1 
            swarm_wf[1, ip] = 0

        # lithosphere
        if swarm_x[ip] > 1000e3 and swarm_z[ip] < Lz - 50e3 and swarm_z[ip] > Lz - 150e3:
            swarm_wf[2, ip] = 1 
            swarm_wf[1, ip] = 0

        # lithosphere
        if swarm_x[ip] > 1000e3 and swarm_x[ip] < 1100e3 and swarm_z[ip] > Lz - 250e3 and swarm_z[ip] < Lz - 50e3:
            swarm_wf[2, ip] = 1 
            swarm_wf[1, ip] = 0

    material_names = ["sticky_air", "mantle", "lithosphere"]

    return swarm_wf, material_names


###################################################################################################
# free slip on all sides


def assign_boundary_conditions_V(
    x_V,
    z_V,
    rad_V,
    theta_V,
    ndof_V,
    Nfem_V,
    nn_V,
    hull_nodes,
    top_nodes,
    bot_nodes,
    left_nodes,
    right_nodes,
):

    bc_fix_V = np.zeros(Nfem_V, dtype=bool)  # boundary condition, yes/no
    bc_val_V = np.zeros(Nfem_V, dtype=np.float64)  # boundary condition, value

    for i in range(0, nn_V):
        if x_V[i] / Lx < eps:
            bc_fix_V[i * ndof_V] = True
            bc_val_V[i * ndof_V] = 0.0
        if x_V[i] / Lx > (1 - eps):
            bc_fix_V[i * ndof_V] = True
            bc_val_V[i * ndof_V] = 0.0
        if z_V[i] / Lz < eps:
            bc_fix_V[i * ndof_V + 1] = True
            bc_val_V[i * ndof_V + 1] = 0.0
        if z_V[i] / Lz > (1 - eps):
            bc_fix_V[i * ndof_V + 1] = True
            bc_val_V[i * ndof_V + 1] = 0.0

    return bc_fix_V, bc_val_V


###################################################################################################

def material_model(
    nparticle,
    swarm_active,
    nmat,
    swarm_wf,
    swarm_x,
    swarm_z,
    swarm_rad,
    swarm_theta,
    swarm_exx,
    swarm_ezz,
    swarm_exz,
    swarm_T,
    swarm_p,
):

    swarm_rho = np.zeros(nparticle, dtype=np.float64)
    swarm_eta = np.zeros(nparticle, dtype=np.float64)
    swarm_hcond = 0 
    swarm_hcapa = 0
    swarm_hprod = 0
    swarm_alpha = 0 
    swarm_mechanism = np.zeros(nparticle, dtype=np.int32)

    swarm_rho = np.zeros(nparticle, dtype=np.float64)
    swarm_eta = np.zeros(nparticle, dtype=np.float64)

    eta_air = 1e19
    rho_air = 0

    eta_mantle = 1e21
    rho_mantle = 3200

    eta_lithosphere = 1e23
    rho_lithosphere = 3300

    for ip in range(0,nparticle):
        if swarm_active[ip]:
           swarm_rho[ip]=swarm_wf[0,ip] * rho_air + swarm_wf[1,ip]* rho_mantle + swarm_wf[2,ip] * rho_lithosphere
           swarm_eta[ip]=swarm_wf[0,ip] * eta_air + swarm_wf[1,ip]* eta_mantle + swarm_wf[2,ip] * eta_lithosphere

    return swarm_rho, swarm_eta, swarm_hcond, swarm_hcapa, swarm_hprod, swarm_alpha, swarm_mechanism


###################################################################################################


def gravity_model(x, z):
    return 0.0, -9.81


###################################################################################################
