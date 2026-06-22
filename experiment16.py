import numpy as np
from constants import *

###################################################################################################

nelx = 100  # 205
nelz = 64  # 93

Lx = 400 * km
Lz = 180 * km

nstep = 1
end_time = 40e6 * year
eta_ref = 1e21
every_solution = 1
every_swarm_vtu = 1

p_scale = 1e6
p_unit = "MPa"
vel_scale = cm / year
vel_unit = "cm/yr"
time_scale = year
time_unit = "yr"

nmat=5
nparticle_per_dim=10
nparticle_min=int(nparticle_per_dim**2*0.85)


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
        if swarm_x[ip] <= 300e3 and swarm_z[ip] >= Lz-10e3:
           swarm_wf[0,ip]=1
        if swarm_x[ip] >= 300e3 and swarm_z[ip] >= Lz-8e3:
           swarm_wf[0,ip]=1
        if swarm_x[ip] <= 300e3 and swarm_z[ip] < Lz - 60e3:
           swarm_wf[1,ip]=1
        if swarm_x[ip] >= 300e3 and swarm_z[ip] < Lz - 18e3:
           swarm_wf[2,ip]=1
        if swarm_x[ip] <= 300e3 and swarm_z[ip] > Lz - 60e3 and swarm_z[ip] <= Lz - 10e3:
           swarm_wf[3,ip]=1
        if swarm_x[ip] >= 300e3 and swarm_z[ip] > Lz - 18e3 and swarm_z[ip] <= Lz - 8e3:
           swarm_wf[4,ip]=1

    material_names = ["water","asthenosphere_L","asthenosphere_R","lithosphere_L","lithosphere_R"]

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

    eta_asthL = 1e21
    rho_asthL = 3200  # asthenosphere left

    eta_asthR = 1e21
    rho_asthR = 3200  # asthenosphere right

    eta_lithL = 1e22
    rho_lithL = 3300  # lithosphere left

    eta_lithR = 1e22
    rho_lithR = 3300  # lithosphere right

    eta_water = 1e18
    rho_water = 1030  # water

    for ip in range(0,nparticle):
        if swarm_active[ip]:
           swarm_rho[ip]=swarm_wf[0,ip] * rho_water +\
                         swarm_wf[1,ip] * rho_asthL +\
                         swarm_wf[2,ip] * rho_asthR +\
                         swarm_wf[3,ip] * rho_lithL +\
                         swarm_wf[4,ip] * rho_lithR
           swarm_eta[ip]=swarm_wf[0,ip] * eta_water +\
                         swarm_wf[1,ip] * eta_asthL +\
                         swarm_wf[2,ip] * eta_asthR +\
                         swarm_wf[3,ip] * eta_lithL +\
                         swarm_wf[4,ip] * eta_lithR

    return swarm_rho, swarm_eta, swarm_hcond, swarm_hcapa, swarm_hprod, swarm_alpha, swarm_mechanism



###################################################################################################


def gravity_model(x, z):
    return 0.0, -9.81


###################################################################################################
