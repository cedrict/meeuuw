import numpy as np
import numba
import scipy
from toolbox import *
from constants import *

###################################################################################################
# Wang & Li, GJI, 2021 - doi: 10.1093/gji/ggab014 (wali21)
###################################################################################################

geometry = "quarter"

nelz = 20

Router = 6370e3
Rinner = Router - 1000e3
Rmean = (Rinner + Router) / 2
top_free_slip = True
bot_free_slip = True

nelx = int(2 * np.pi * Rmean / 4 / (Router - Rinner) * nelz)

debug_ascii = False

solve_T = True
Tsurf = 273
Tcmb = 1350 + 273

time_scale = year
time_unit = "yr"
vel_scale = cm / year
vel_unit = "cm/yr"
p_scale = 1e6
p_unit = "MPa"

alphaT = 3e-5
T0 = Tsurf
kappa = 1e-6
hcapa0 = 1250
rho0 = 3300
hcond0 = kappa * rho0 * hcapa0
eta0 = 1e21
Rp = 300e3
DTp = 300

end_time = 1000e6 * year
every_solution = 10
every_swarm_vtu = 10
RKorder = -1

nstep = 10000

eta_ref = 1e22

###############################################################################


def initial_temperature(x, z, rad, theta, nn_V):

    T = np.zeros(nn_V, dtype=np.float64)

    age_surf = 80e6 * year
    age_cmb = 80e6 * year  # irrelevant

    Tm = Tcmb

    for i in range(0, nn_V):
        T[i] = initial_temperature_hsc(rad[i], Rinner, Router, Tcmb, Tsurf, age_cmb, age_surf, Tm, kappa)

    return T


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

    for i in range(0, nn_V):
        # left vertical wall
        if x_V[i] / Rinner < eps:
            bc_fix_V[i * ndof_V] = True
            bc_val_V[i * ndof_V] = 0.0
        # horizontal wall
        if z_V[i] / Rinner < eps:
            bc_fix_V[i * ndof_V + 1] = True
            bc_val_V[i * ndof_V + 1] = 0.0
        # top and bottom
        if not bot_free_slip and bot_nodes[i]:
            bc_fix_V[i * ndof_V] = True
            bc_val_V[i * ndof_V] = 0.0  # no slip
            bc_fix_V[i * ndof_V + 1] = True
            bc_val_V[i * ndof_V + 1] = 0.0
        if not top_free_slip and top_nodes[i]:
            bc_fix_V[i * ndof_V] = True
            bc_val_V[i * ndof_V] = 0.0  # no slip
            bc_fix_V[i * ndof_V + 1] = True
            bc_val_V[i * ndof_V + 1] = 0.0
        # pin all four corners to u=w=0
        if left_nodes[i] and bot_nodes[i]:
            bc_fix_V[i * ndof_V] = True
            bc_val_V[i * ndof_V] = 0.0
            bc_fix_V[i * ndof_V + 1] = True
            bc_val_V[i * ndof_V + 1] = 0.0
        if right_nodes[i] and bot_nodes[i]:
            bc_fix_V[i * ndof_V] = True
            bc_val_V[i * ndof_V] = 0.0
            bc_fix_V[i * ndof_V + 1] = True
            bc_val_V[i * ndof_V + 1] = 0.0
        if left_nodes[i] and top_nodes[i]:
            bc_fix_V[i * ndof_V] = True
            bc_val_V[i * ndof_V] = 0.0
            bc_fix_V[i * ndof_V + 1] = True
            bc_val_V[i * ndof_V + 1] = 0.0
        if right_nodes[i] and top_nodes[i]:
            bc_fix_V[i * ndof_V] = True
            bc_val_V[i * ndof_V] = 0.0
            bc_fix_V[i * ndof_V + 1] = True
            bc_val_V[i * ndof_V + 1] = 0.0

    return bc_fix_V, bc_val_V


###############################################################################


def assign_boundary_conditions_T(
    x_T,
    z_T,
    rad_T,
    theta_T,
    Nfem_T,
    nn_T,
    hull_nodes,
    top_nodes,
    bot_nodes,
    left_nodes,
    right_nodes,
):

    bc_fix_T = np.zeros(Nfem_T, dtype=bool)
    bc_val_T = np.zeros(Nfem_T, dtype=np.float64)

    for i in range(0, nn_T):
        if bot_nodes[i]:
            dist2 = (x_T[i] - Rinner / 1.414) ** 2 + (z_T[i] - Rinner / 1.414) ** 2
            bc_fix_T[i] = True
            bc_val_T[i] = Tcmb + DTp * np.exp(-dist2 / Rp**2)
        if top_nodes[i]:
            bc_fix_T[i] = True
            bc_val_T[i] = Tsurf

    return bc_fix_T, bc_val_T


###############################################################################


def particle_layout(nparticle, nmat, swarm_x, swarm_z, swarm_rad, swarm_theta, Lx, Lz):

    swarm_wf = np.zeros((nmat, nparticle), dtype=np.float64)
    swarm_wf[:, :] = 1

    material_names = ["mantle"]

    return swarm_wf, material_names


###############################################################################

@numba.njit
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
    swarm_hcond = np.zeros(nparticle, dtype=np.float64)
    swarm_hcapa = np.zeros(nparticle, dtype=np.float64)
    swarm_hprod = np.zeros(nparticle, dtype=np.float64)
    swarm_alpha = np.zeros(nparticle, dtype=np.float64)
    swarm_mechanism = np.zeros(nparticle, dtype=np.int32)

    for i in range(nparticle):
        swarm_eta[i] = eta0 * np.exp(6.91 * (1 - (swarm_T[i] - 273) / 1350))
        swarm_eta[i] = min(swarm_eta[i], 1e24)
        swarm_eta[i] = max(swarm_eta[i], 1e19)

    swarm_mechanism[:] = 1
    swarm_alpha[:] = alphaT
    swarm_hcond[:] = hcond0
    swarm_hcapa[:] = hcapa0
    swarm_hprod[:] = 4e-12 * rho0
    swarm_rho[:] = rho0 * (1 - swarm_alpha[:] * (swarm_T[:] - T0))

    return swarm_rho, swarm_eta, swarm_hcond, swarm_hcapa, swarm_hprod, swarm_alpha, swarm_mechanism


###############################################################################


@numba.njit
def gravity_model(x, z):
    g0 = 9.8
    gx = -x / np.sqrt(x**2 + z**2) * g0
    gz = -z / np.sqrt(x**2 + z**2) * g0
    return gx, gz


###############################################################################
