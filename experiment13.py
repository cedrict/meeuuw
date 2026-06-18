import numpy as np
import numba
from constants import *

###################################################################################################

nelz = 32
nelx = 32

Lx = 512 * km
Lz = 512 * km

nmat = 2
nstep = 1
eta_ref = 1e22
p_scale = 1e6
p_unit = "MPa"
vel_scale = cm / year
vel_unit = "cm/yr"
time_scale = year
time_unit = "yr"
every_solution = 1
every_swarm_vtu = 1
nparticle_per_dim = 6
# averaging='arithmetic'
# averaging='geometric'
averaging = "harmonic"
debug_ascii = True
debug_solver = True
end_time = 120e6 * year

eta_mantle = 1e21
rho_mantle = 3200
eta_block = 1e21
rho_block = 3208

nsamplepoints = 1
xsamplepoints = [256e3]
zsamplepoints = [384e3]

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

    eps = 1e-8

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

    material_names = ["mantle", "block"]

    for ip in range(0, nparticle):
        if abs(swarm_x[ip] - Lx / 2) < 64e3 and abs(swarm_z[ip] - 384e3) < 64e3:
            swarm_wf[1, ip] = 1
        else:
            swarm_wf[0, ip] = 1

    return swarm_wf, material_names


###################################################################################################


@numba.njit
def material_model(
    nparticle,
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

    for ip in range(0, nparticle):
        swarm_rho[ip] = swarm_wf[0, ip] * rho_mantle + swarm_wf[1, ip] * rho_block
        swarm_eta[ip] = swarm_wf[0, ip] * eta_mantle + swarm_wf[1, ip] * eta_block

    # mask=(swarm_mat==1) ; swarm_eta[mask]=eta_mantle ; swarm_rho[mask]=rho_mantle
    # mask=(swarm_mat==2) ; swarm_eta[mask]=eta_block  ; swarm_rho[mask]=rho_block

    return swarm_rho, swarm_eta, swarm_hcond, swarm_hcapa, swarm_hprod


###################################################################################################


def gravity_model(x, z):
    return 0.0, -10.0


###################################################################################################
