import numpy as np
from constants import *

###################################################################################################

Lx = 1400e3  # half domain!
Lz = 700e3
eta_ref = 1e21

p_scale = 1e6
p_unit = "MPa"
vel_scale = cm / year
vel_unit = "cm/yr"
time_scale = year
time_unit = "yr"

CFLnb=0.
pressure_normalisation = "none"

nmat=2

#use_free_surface=True

# a: cosine perturbation
# b: plume

icase = "a"

if icase == "a":
    nelx = 14
    nelz = 7
    nstep = 1
    end_time = 2e5 * year

if icase == "b":
    nelx = 140
    nelz = 100
    nstep = 1500
    end_time = 20e6 * year

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
            bc_fix_V[i * ndof_V] = True
            bc_val_V[i * ndof_V] = 0.0
            bc_fix_V[i * ndof_V + 1] = True
            bc_val_V[i * ndof_V + 1] = 0.0

    return bc_fix_V, bc_val_V


###################################################################################################


def particle_layout(nparticle, nmat, swarm_x, swarm_z, swarm_rad, swarm_theta, Lx, Lz):

    swarm_wf = np.zeros((nmat, nparticle), dtype=np.float64)

    if icase == "a":
        for ip in range(0, nparticle):
            if swarm_z[ip] > 600e3:
               swarm_wf[0, ip] = 1
            else:
               swarm_wf[1, ip] = 1

    #if icase == "b":
    #    for ip in range(0, nparticle):
    #        if swarm_z[ip] > 600e3:
    #            swarm_mat[ip] = 1  # lithosphere
    #        if swarm_z[ip] > 700e3:
    #            swarm_mat[ip] = 0  # sticky air
    #        if (swarm_x[ip] - Lx) ** 2 + (swarm_z[ip] - 300e3) ** 2 < 50e3**2:
    #            swarm_mat[ip] = 3  # plume

    material_names = ["lithosphere", "mantle"]

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

    swarm_rho[:]=3300

    for ip in range(0, nparticle):
        if swarm_active[ip]:
           swarm_eta[ip]=swarm_wf[0,ip] * 1e23 + swarm_wf[1,ip]* 1e21
          
    return swarm_rho, swarm_eta, swarm_hcond, swarm_hcapa, swarm_hprod, swarm_alpha, swarm_mechanism


###################################################################################################


def gravity_model(x, z):
    return 0.0, -10.0


###################################################################################################
