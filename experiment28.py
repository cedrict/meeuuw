import numpy as np
import numba
from constants import *

###################################################################################################
# setup based on gops27
# "The numerical (width) and (depth) resolution is 201 × 101 Eulerian nodes and
# 601 × 301 Lagrangian nodes. Half of the Eulerian and Lagrangian elements are
# concentrated in the top 160 km in order to enhance resolution in the lithosphere.
# The model has a free top surface, allowing topography to develop as the model
# evolves. The mechanical boundary conditions at the other three sides are deﬁned
# by zero tangential stress and normal velocity (e.g., free slip)."
# -> this means 3x3 particles by element at startup...

# min/max viscosity?

nelx = 150
nelz = 75

Lx = 2000 * km
Lz = 1000 * km

nstep = 1
dt_max = 50000 * year
end_time = 100e6 * year
eta_ref = 1e21
every_solution = 1
every_swarm_vtu = 10

p_scale = 1e6
p_unit = "MPa"
vel_scale = cm / year
vel_unit = "cm/yr"
time_scale = year
time_unit = "yr"

nmat = 3
nparticle_per_dim = 10
particle_distribution = 3  # 0: random, 1: reg, 2: Poisson Disc, 3: pseudo-random

solve_T = True
Tbottom = 1525 + 273
Ttop = 25 + 273

rho0 = 3300

nonlinear = True
niter_nl = 5
tol_nl = 1e-2

use_stretching_x=True
use_stretching_z=True
n_segments_x=4
n_segments_z=3
x_segments=np.array([0,0.2,0.5,0.8,1], dtype=np.float64)
z_segments=np.array([0,0.5,0.8,1], dtype=np.float64)
nelx_segments=np.array([25,50,50,25], dtype=np.int16)
nelz_segments=np.array([30,30,30], dtype=np.int16)
nelx=nelx_segments.sum()
nelz=nelz_segments.sum()


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
        if swarm_z[ip] >= Lz - 41.5e3:  # crust
            swarm_wf[0, ip] = 1
        elif swarm_z[ip] >= Lz - 160e3 and swarm_z[ip] < Lz - 41.5e3:  # arc root
            swarm_wf[1, ip] = 1
        elif abs(swarm_x[ip] - Lx / 2) < 140e3 and swarm_z[ip] >= Lz - 201.5e3 and swarm_z[ip] < Lz - 160e3:  # arc root
            swarm_wf[1, ip] = 1
        else:  # mantle
            swarm_wf[2, ip] = 1

    material_names = ["crust", "arc_root", "mantle"]

    return swarm_wf, material_names


###################################################################################################


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

    swarm_alpha[:] = 2e-5
    swarm_hcond[:] = 2.25
    swarm_hcapa[:] = 1250

    for ip in range(0, nparticle):
        if swarm_active[ip]:
            if swarm_wf[0, ip] > 0.99:  # crust
                rho = 2840
                A = 1.1e-28
                n = 4
                Q = 223e3
                c = 1e6
                phi = 15 / 180 * np.pi
            if swarm_wf[1, ip] > 0.99:  # root
                rho = 3300
                A = 1e-38
                n = 3.5
                Q = 0
                c = 1e20
                phi = 0
            if swarm_wf[2, ip] > 0.99:  # mantle
                rho = 3260
                A = 4.89e-17
                n = 3.5
                Q = 535e3
                c = 1e20
                phi = 0

            swarm_rho[ip] = rho * (1 - swarm_alpha[ip] * (swarm_T[ip]-298))

            # compute effective strain rate
            e = np.sqrt(0.5 * (swarm_exx[ip] ** 2 + swarm_ezz[ip] ** 2) + swarm_exz[ip] ** 2)
            e = min(e, 1e-12)
            e = max(e, 1e-20)

            # compute effective dislocation creep viscosity
            eta_eff = 0.5 * A ** (-1 / n) * e ** (1 / n - 1) * np.exp(Q / (n * Rgas * swarm_T[ip]))

            # implement stress limiter ('Drucker-Prager plasticity')
            sigma_y = swarm_p[ip] * np.sin(phi) + c * np.cos(phi)
            if 2 * eta_eff * e > sigma_y:
                eta_eff = sigma_y / 2 / e
                swarm_mechanism[ip] = 1

            eta_eff = max(5e19, eta_eff)
            eta_eff = min(5e22, eta_eff)

            swarm_eta[ip] = eta_eff

    return swarm_rho, swarm_eta, swarm_hcond, swarm_hcapa, swarm_hprod, swarm_alpha, swarm_mechanism


###################################################################################################


def assign_boundary_conditions_T(
    x_T, z_T, rad_T, theta_T, Nfem_T, nn_T, hull_Tnodes, top_Tnodes, bot_Tnodes, left_Tnodes, right_Tnodes
):

    eps = 1e-8

    bc_fix_T = np.zeros(Nfem_T, dtype=bool)
    bc_val_T = np.zeros(Nfem_T, dtype=np.float64)

    for i in range(0, nn_T):
        if z_T[i] < eps:
            bc_fix_T[i] = True
            bc_val_T[i] = Tbottom
        if z_T[i] > (Lz - eps):
            bc_fix_T[i] = True
            bc_val_T[i] = Ttop

    return bc_fix_T, bc_val_T


###################################################################################################


def initial_temperature(x, z, rad, theta, nn_V):

    T = np.zeros(nn_V, dtype=np.float64)

    z3 = 1000e3
    z2 = 1000e3 - 41.5e3
    z1 = 840e3
    z0 = 0

    T3 = 25 + 273
    T2 = 550 + 273
    T1 = 1350 + 273
    T0 = 1525 + 273

    for ip in range(0, nn_V):
        if z[ip] <= z1:
            T[ip] = (z[ip] - z0) / (z1 - z0) * (T1 - T0) + T0
        elif z[ip] <= z2:
            T[ip] = (z[ip] - z1) / (z2 - z1) * (T2 - T1) + T1
        elif z[ip] <= z3:
            T[ip] = (z[ip] - z2) / (z3 - z2) * (T3 - T2) + T2

    return T


###################################################################################################


def gravity_model(x, z):
    return 0.0, -9.81


###################################################################################################
