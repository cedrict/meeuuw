import numpy as np
from constants import *

###################################################################################################

#axisymmetric = True
# straighten_edges=True
#remove_rho_profile = True
# method_nodal_strain_rate=2
#mapping = "Q1"

# geometry='quarter'
geometry = "half"

nelz = 64

if geometry == "quarter":
    nelx = int(3 * nelz)
# if geometry=='half': nelx=int(6.7*nelz)
if geometry == "half":
    nelx = int(6.0 * nelz)

Rinner = 3400e3
Router = 6400e3

nmat=2
eta_blob = 1e21
rho_blob = 3960
depth_blob = 1500e3
radius_blob = 400e3

eta_mantle = 1e21
rho_mantle = 4000

compute_dynamic_topography = True
rho_DT_top = 0
rho_DT_bot = 10000

nstep = 1
eta_ref = 1e21
p_scale = 1e6
p_unit = "MPa"
vel_scale = cm / year
vel_unit = "cm/yr"
time_scale = year
time_unit = "yr"
every_solution = 1
every_swarm_vtu = 1
particle_distribution = 1  # 0: random, 1: reg, 2: Poisson Disc, 3: pseudo-random
debug_ascii = True
CFLnb = 0.0
end_time = 100e6 * year

top_free_slip = True
bot_free_slip = True

gravity_npts = 0
gravity_height = 200e3
gravity_rho_ref = 0

###############################################################################


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

    if geometry == "quarter":
        for i in range(0, nn_V):
            if x_V[i] / Rinner < eps: # left boundary
                bc_fix_V[i * ndof_V] = True 
                bc_val_V[i * ndof_V] = 0.0
            if z_V[i] / Rinner < eps: # bottom boundary
                bc_fix_V[i * ndof_V + 1] = True
                bc_val_V[i * ndof_V + 1] = 0.0
            if bot_nodes[i] and right_nodes[i]: # corner
                bc_fix_V[i * ndof_V] = True
                bc_val_V[i * ndof_V] = 0.0
                bc_fix_V[i * ndof_V + 1] = True
                bc_val_V[i * ndof_V + 1] = 0.0
            if top_nodes[i] and right_nodes[i]: # corner
                bc_fix_V[i * ndof_V] = True
                bc_val_V[i * ndof_V] = 0.0
                bc_fix_V[i * ndof_V + 1] = True
                bc_val_V[i * ndof_V + 1] = 0.0
            if bot_nodes[i] and left_nodes[i]: # corner
                bc_fix_V[i * ndof_V] = True
                bc_val_V[i * ndof_V] = 0.0
                bc_fix_V[i * ndof_V + 1] = True
                bc_val_V[i * ndof_V + 1] = 0.0
            if top_nodes[i] and left_nodes[i]: # corner
                bc_fix_V[i * ndof_V] = True 
                bc_val_V[i * ndof_V] = 0.0
                bc_fix_V[i * ndof_V + 1] = True
                bc_val_V[i * ndof_V + 1] = 0.0

    if geometry == "half":
        for i in range(0, nn_V):
            if x_V[i] / Rinner < eps: # left vertical
                bc_fix_V[i * ndof_V] = True
                bc_val_V[i * ndof_V] = 0.0
            if bot_nodes[i] and left_nodes[i]: # corner
                bc_fix_V[i * ndof_V] = True
                bc_val_V[i * ndof_V] = 0.0
                bc_fix_V[i * ndof_V + 1] = True
                bc_val_V[i * ndof_V + 1] = 0.0
            if bot_nodes[i] and right_nodes[i]: # corner
                bc_fix_V[i * ndof_V] = True
                bc_val_V[i * ndof_V] = 0.0
                bc_fix_V[i * ndof_V + 1] = True
                bc_val_V[i * ndof_V + 1] = 0.0
            if top_nodes[i] and left_nodes[i]: # corner
                bc_fix_V[i * ndof_V] = True
                bc_val_V[i * ndof_V] = 0.0
                bc_fix_V[i * ndof_V + 1] = True
                bc_val_V[i * ndof_V + 1] = 0.0
            if top_nodes[i] and right_nodes[i]: # corner
                bc_fix_V[i * ndof_V] = True
                bc_val_V[i * ndof_V] = 0.0
                bc_fix_V[i * ndof_V + 1] = True
                bc_val_V[i * ndof_V + 1] = 0.0
            #if bot_nodes[i]:
            #    bc_fix_V[i * ndof_V] = True
            #    bc_val_V[i * ndof_V] = 0.0
            #    bc_fix_V[i * ndof_V + 1] = True
            #    bc_val_V[i * ndof_V + 1] = 0.0

    return bc_fix_V, bc_val_V


###################################################################################################


def particle_layout(nparticle, nmat, swarm_x, swarm_z, swarm_rad, swarm_theta, Lx, Lz):

    swarm_wf = np.zeros((nmat, nparticle), dtype=np.float64)

    for ip in range(nparticle):
        if (swarm_x[ip]) ** 2 + (swarm_z[ip] - (Router - depth_blob)) ** 2 < radius_blob**2:
           swarm_wf[1, ip] = 1
        else:
           swarm_wf[0, ip] = 1

    material_names = ["mantle", "blob"]

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
    swarm_alpha = 0 
    swarm_mechanism = np.zeros(nparticle, dtype=np.int32)

    for ip in range(0,nparticle):
        if swarm_active[ip]:
           swarm_rho[ip]=swarm_wf[0,ip] * rho_mantle +\
                         swarm_wf[1,ip] * rho_blob 
           swarm_eta[ip]=swarm_wf[0,ip] * eta_mantle +\
                         swarm_wf[1,ip] * eta_blob 

    return swarm_rho, swarm_eta, swarm_hcond, swarm_hcapa, swarm_hprod, swarm_alpha, swarm_mechanism


###################################################################################################


def gravity_model(x, z):
    g0 = 10
    gx = -x / np.sqrt(x**2 + z**2) * g0
    gz = -z / np.sqrt(x**2 + z**2) * g0
    return gx, gz


###################################################################################################
