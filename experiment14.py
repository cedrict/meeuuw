import numpy as np
from constants import *

###################################################################################################

nelx = 100
nelz = 66

Lx = 1000 * km
Lz = 660 * km

CFLnb = 0.01
nstep = 100
eta_ref = 1e22
p_scale = 1e6
p_unit = "MPa"
vel_scale = cm / year
vel_unit = "cm/yr"
time_scale = year
time_unit = "yr"
every_solution = 1
every_swarm_vtu = 1
every_swarm_png = 10
every_quadpoints_vtu = 10
end_time = 120e6 * year
dt_max = 1e4 * year
nmat=2

icase = 2

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
            bc_fix_V[i * ndof_V + 1] = True
            bc_val_V[i * ndof_V + 1] = 0.0
        if x_V[i] / Lx > (1 - eps):
            bc_fix_V[i * ndof_V] = True
            bc_val_V[i * ndof_V] = 0.0
            bc_fix_V[i * ndof_V + 1] = True
            bc_val_V[i * ndof_V + 1] = 0.0
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
        if (swarm_z[ip] > 660e3 - 80e3 and swarm_z[ip] <= 660e3) or (
            swarm_z[ip] > 660e3 - (80e3 + 250e3) and abs(swarm_x[ip] - Lx / 2) < 40e3):
            swarm_wf[1,ip] = 1
        else:
            swarm_wf[0,ip] = 1

    material_names = ["mantle", "lithosphere"]

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

    eta_mantle = 1e21
    rho_mantle = 3150
    eta_lithosphere = 1e22
    rho_lithosphere = 3300

    if icase == 0:
        for ip in range(0, nparticle):
            if swarm_active[ip]:
               swarm_rho[ip] = swarm_wf[0, ip] * rho_mantle + swarm_wf[1, ip] * rho_lithosphere
               swarm_eta[ip] = swarm_wf[0, ip] * eta_mantle + swarm_wf[1, ip] * eta_lithosphere


    if icase == 1:
        swarm_sr = np.sqrt(0.5 * (swarm_exx**2 + swarm_ezz**2) + swarm_exz**2)
        for ip in range(0, nparticle):
            if swarm_active[ip]:
               swarm_rho[ip] = swarm_wf[0, ip] * rho_mantle + swarm_wf[1, ip] * rho_lithosphere
               if swarm_wf[0,ip]>0.99:
                  swarm_eta[ip] = 1e21
               else:
                  sr = max(1e-30, swarm_sr[ip])
                  n_pow = 4
                  val = (4.75e11) * sr ** (1.0 / n_pow - 1.0)
                  val = max(val, 1e19)
                  val = min(val, 1e25)
                  swarm_eta[ip] = val

    if icase == 2:
        swarm_sr = np.sqrt(0.5 * (swarm_exx**2 + swarm_ezz**2) + swarm_exz**2)
        for ip in range(0, nparticle):
            if swarm_active[ip]:
               swarm_rho[ip] = swarm_wf[0, ip] * rho_mantle + swarm_wf[1, ip] * rho_lithosphere
               sr = max(1e-30, swarm_sr[ip])
               if swarm_wf[0,ip] > 0.99:
                   val = (4.54e10) * sr ** (1.0 / 3 - 1.0)
               else:
                   val = (4.75e11) * sr ** (1.0 / 4 - 1.0)
               val = max(val, 1e19)
               val = min(val, 1e25)
               swarm_eta[ip] = val

    return swarm_rho, swarm_eta, swarm_hcond, swarm_hcapa, swarm_hprod, swarm_alpha, swarm_mechanism


###################################################################################################


def gravity_model(x, z):
    return 0.0, -10.0


###################################################################################################
