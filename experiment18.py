import numpy as np
from constants import *

###################################################################################################

nelx = 64
nelz = 64

Lx = 1
Lz = 1

nstep = 1
end_time = 0
eta_ref = 1
every_solution = 1
every_swarm_vtu = 1

nmat=3

###################################################################################################


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

def particle_layout(nparticle, nmat, swarm_x, swarm_z, swarm_rad, swarm_theta, Lx, Lz):

    swarm_wf = np.zeros((nmat, nparticle), dtype=np.float64)

    for ip in range(0, nparticle):
        if swarm_z[ip]>= 0.75: # air
           swarm_wf[0,ip]=1
        elif (swarm_x[ip] - 0.5) ** 2 + (swarm_z[ip] - 0.6) ** 2 < 0.123456789**2: # sphere
           swarm_wf[2,ip]=1
        else: # mantle
           swarm_wf[1,ip]=1

    material_names = ["air","mantle","sphere"]

    return swarm_wf, material_names


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
    swarm_alpha = np.zeros(nparticle, dtype=np.float64)
    swarm_mechanism = np.zeros(nparticle, dtype=np.int32)

    rho_air=0
    rho_mantle=1
    rho_sphere=2

    eta_air=1e-3
    eta_mantle=1
    eta_sphere=1e3

    for ip in range(0,nparticle):
        if swarm_active[ip]:
           swarm_rho[ip]=swarm_wf[0,ip] * rho_air + swarm_wf[1,ip]* rho_mantle + swarm_wf[2,ip] * rho_sphere
           swarm_eta[ip]=swarm_wf[0,ip] * eta_air + swarm_wf[1,ip]* eta_mantle + swarm_wf[2,ip] * eta_sphere

    return swarm_rho, swarm_eta, swarm_hcond, swarm_hcapa, swarm_hprod, swarm_alpha, swarm_mechanism


###################################################################################################


def gravity_model(x, z):
    return 0.0, -1.0


###################################################################################################
