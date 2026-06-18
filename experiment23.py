import numpy as np
import numba
import scipy
from toolbox import *
from constants import *

###################################################################################################
# Coltice et al , Science advances, 2019 (aso Arnould et al, G3, 2018)

# TODO: plasticity

# geometry='box'
# geometry='eighth'
geometry = "quarter"
# geometry='half'
# geometry='annulus'

nelz = 28

match geometry:
    case "box":
        Lx = 4500e3
        Lz = 3000e3
        nelx = int(Lx / Lz * nelz)
    case "eighth":
        Router = 6370e3
        Rinner = Router - 3000e3
        Rmean = (Rinner + Router) / 2
        nelx = int(2 * np.pi * Rmean / 8 / (Router - Rinner) * nelz)
        top_free_slip = True
        bot_free_slip = True
    case "quarter":
        Router = 6370e3
        Rinner = Router - 3000e3
        Rmean = (Rinner + Router) / 2
        nelx = int(2 * np.pi * Rmean / 4 / (Router - Rinner) * nelz)
        top_free_slip = True
        bot_free_slip = True
    case "half":
        Router = 6370e3
        Rinner = Router - 3000e3
        Rmean = (Rinner + Router) / 2
        nelx = int(2 * np.pi * Rmean / 2 / (Router - Rinner) * nelz)
        top_free_slip = True
        bot_free_slip = True
    case "annulus":
        Router = 6370e3
        Rinner = Router - 3000e3
        Rmean = (Rinner + Router) / 2
        nelx = int(2 * np.pi * Rmean / 1 / (Router - Rinner) * nelz)
        top_free_slip = True
        bot_free_slip = True

debug_ascii = False

solve_T = True
Tsurf = 273
Tcmb = 2390

time_scale = year
time_unit = "yr"
vel_scale = cm / year
vel_unit = "cm/yr"
p_scale = 1e6
p_unit = "MPa"

alphaT = 3e-5
T0 = Tsurf
hcond0 = 3.15
hcapa0 = 716
rho0 = 4400
kappa = 1e-6

end_time = 5000e6 * year
every_solution = 10
every_swarm_vtu = 10
RKorder = -1

nstep = 10000

eta_ref = 1e22

CFLnb = 0.75

###############################################################################


def initial_temperature(x, z, rad, theta, nn_V):

    T = np.zeros(nn_V, dtype=np.float64)

    age_surf = 250e6 * year
    age_cmb = 250e6 * year

    coeff = 0.7
    Tm = Tsurf + (Tcmb - Tsurf) * coeff

    match geometry:
        case "box":
            for i in range(0, nn_V):
                T[i] = initial_temperature_hsc(z[i], 0, Lz, Tcmb, Tsurf, age_cmb, age_surf, Tm, kappa)
                T[i] += (
                    0.010 * Tm * np.cos(3 * np.pi * x[i] / Lx) * np.sin(5 * np.pi * z[i] / Lz)
                    + 0.015 * Tm * np.cos(4 * np.pi * x[i] / Lx) * np.sin(4 * np.pi * z[i] / Lz)
                    + 0.005 * Tm * np.cos(5 * np.pi * x[i] / Lx) * np.sin(3 * np.pi * z[i] / Lz)
                )

        case "quarter" | "half" | "eighth" | "annulus":
            for i in range(0, nn_V):
                T[i] = initial_temperature_hsc(rad[i], Rinner, Router, Tcmb, Tsurf, age_cmb, age_surf, Tm, kappa)
                T[i] += (
                    0.01 * Tm * np.sin(3 * theta[i])
                    + 0.02 * Tm * np.sin(7 * theta[i])
                    + 0.03 * Tm * np.sin(9 * theta[i])
                )

    return T


###############################################################################
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

    match geometry:
        case "box":
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

        # case 'eighth' :
        #  for i in range(0,nn_V):
        #      if left_nodes[i]:
        #         bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
        #      if right_nodes[i]:
        #         bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0. # no slip
        #         bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
        #      if bot_nodes[i]:
        #         bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0. # no slip
        #         bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
        #      if top_nodes[i]:
        #         bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0. # no slip
        #         bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

        case "eighth" | "quarter" | "half":
            for i in range(0, nn_V):
                if x_V[i] / Rinner < eps:
                    bc_fix_V[i * ndof_V] = True
                    bc_val_V[i * ndof_V] = 0.0
                if geometry == "quarter" and z_V[i] / Rinner < eps:
                    bc_fix_V[i * ndof_V + 1] = True
                    bc_val_V[i * ndof_V + 1] = 0.0
                if geometry == "eighth" and right_nodes[i]:
                    bc_fix_V[i * ndof_V] = True
                    bc_val_V[i * ndof_V] = 0.0  # no slip
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
                    bc_val_V[i * ndof_V] = 0.0  # no slip
                    bc_fix_V[i * ndof_V + 1] = True
                    bc_val_V[i * ndof_V + 1] = 0.0
                if right_nodes[i] and bot_nodes[i]:
                    bc_fix_V[i * ndof_V] = True
                    bc_val_V[i * ndof_V] = 0.0  # no slip
                    bc_fix_V[i * ndof_V + 1] = True
                    bc_val_V[i * ndof_V + 1] = 0.0
                if left_nodes[i] and top_nodes[i]:
                    bc_fix_V[i * ndof_V] = True
                    bc_val_V[i * ndof_V] = 0.0  # no slip
                    bc_fix_V[i * ndof_V + 1] = True
                    bc_val_V[i * ndof_V + 1] = 0.0
                if right_nodes[i] and top_nodes[i]:
                    bc_fix_V[i * ndof_V] = True
                    bc_val_V[i * ndof_V] = 0.0  # no slip
                    bc_fix_V[i * ndof_V + 1] = True
                    bc_val_V[i * ndof_V + 1] = 0.0

        case "annulus":
            for i in range(0, nn_V):
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

    return bc_fix_V, bc_val_V


###############################################################################


def assign_boundary_conditions_T(
    x_V,
    z_V,
    rad_V,
    theta_V,
    Nfem_T,
    nn_V,
    hull_nodes,
    top_nodes,
    bot_nodes,
    left_nodes,
    right_nodes,
):

    bc_fix_T = np.zeros(Nfem_T, dtype=bool)
    bc_val_T = np.zeros(Nfem_T, dtype=np.float64)

    for i in range(0, nn_V):
        if bot_nodes[i]:
            bc_fix_T[i] = True
            bc_val_T[i] = Tcmb
        if top_nodes[i]:
            bc_fix_T[i] = True
            bc_val_T[i] = Tsurf

    return bc_fix_T, bc_val_T


###############################################################################


def particle_layout(nparticle, nmat, swarm_x, swarm_z, swarm_rad, swarm_theta, Lx, Lz):

    swarm_wf = np.zeros((nmat, nparticle), dtype=np.floa64)
    swarm_wf[:, :] = 1

    material_names = ["mantle"]

    return swarm_wf, material_names


###############################################################################
# Coltice private communication: "je me suis rendu compte que j'ai fait une erreur
# dans le redimensionnement du volume d'activation dans le modèle de 2019. Dans le
# sup. mat. tu trouveras 13.8cm3/mol alors qu'en refaisant les calculs on trouve
# en réalité 0.7cm3/mol. Dans une exponentielle, ça fait un petit paquet...
# Donc si tu tentes de refaire la simu, fais gaffe à ça".
# Note that Arnould et al use the lithostatic pressure in the viscosity
# I have here implemented a linear diminution of the thermal expansion coeff
# of a factor 3 (from 3e-5 at the top to 1e-5 at the bottom) as in the
# Coltice paper, although their profile is data-driven and therefore
# probably not linear...


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
    swarm_hcond = np.zeros(nparticle, dtype=np.float64)
    swarm_hcapa = np.zeros(nparticle, dtype=np.float64)
    swarm_hprod = np.zeros(nparticle, dtype=np.float64)
    swarm_alpha = np.zeros(nparticle, dtype=np.float64)
    swarm_mechanism = np.zeros(nparticle, dtype=np.int32)

    E_a = 160e3
    V_a = 0.2e-6  # 13.8e-6
    T_0 = 1530
    djump = 100e3 / 2
    alpha1 = 1e-5
    alpha2 = 2e-5
    alpha3 = 2.5e-5
    alpha4 = 4e-5
    sigma_y = 61e6
    min_strainrate = 1e-21

    match geometry:
        case "box":
            zjump = Lz - 660e3
            for i in range(nparticle):
                swarm_eta[i] = (
                    1e21
                    * np.exp(np.log(30) * (1 - 0.5 * (1 - np.tanh((zjump - swarm_z[i]) / djump))))
                    * np.exp((E_a + swarm_p[i] * V_a) / Rgas / swarm_T[i] - E_a / Rgas / T_0)
                )
                swarm_eta[i] = min(swarm_eta[i], 1e25)
                swarm_eta[i] = max(swarm_eta[i], 1e20)
                # alpha_i=alphaT*(1-(Lz-swarm_z[i])/Lz*2./3) linear bottom-top
                if swarm_z[i] > zjump:  # upper mantle
                    alpha_i = (alpha4 - alpha2) / (Lz - zjump) * (swarm_z[i] - zjump) + alpha2
                else:
                    alpha_i = (alpha3 - alpha1) / (zjump) * swarm_z[i] + alpha1
                swarm_rho[i] = rho0 * (1 - alpha_i * (swarm_T[i] - T0))
                swarm_alpha[i] = alpha_i
                # print(swarm_z[i],alpha_i)
        case "quarter" | "half" | "eighth" | "annulus":
            Rjump = Router - 660e3
            for i in range(nparticle):
                eff_strainrate = np.sqrt(0.5 * (swarm_exx[i] ** 2 + swarm_ezz[i] ** 2) + swarm_exz[i] ** 2)
                eff_strainrate = max(min_strainrate, eff_strainrate)

                # plasticity only present in lithosphere ~ 120km thick
                if swarm_rad[i] < Router - 120e3:
                    eta_p = 1e40
                else:
                    eta_p = (sigma_y + 1000 * (Router - swarm_rad[i])) / 2 / eff_strainrate

                eta_v = (
                    1e21
                    * np.exp(np.log(30) * (1 - 0.5 * (1 - np.tanh((Rjump - swarm_rad[i]) / djump))))
                    * np.exp((E_a + swarm_p[i] * V_a) / Rgas / swarm_T[i] - E_a / Rgas / T_0)
                )

                if eta_v < eta_p:
                    swarm_eta[i] = eta_v
                    swarm_mechanism[i] = 1
                else:
                    swarm_eta[i] = eta_p
                    swarm_mechanism[i] = 2

                swarm_eta[i] = min(swarm_eta[i], 1e25)
                swarm_eta[i] = max(swarm_eta[i], 1e19)
                # alpha_i=alphaT*(1.-(Router-swarm_rad[i])/(Router-Rinner)*2./3.) # linear top-botom
                if swarm_rad[i] > Rjump:  # upper mantle
                    alpha_i = (alpha4 - alpha2) / (Router - Rjump) * (swarm_rad[i] - Rjump) + alpha2
                else:
                    alpha_i = (alpha3 - alpha1) / (Rjump - Rinner) * (swarm_rad[i] - Rinner) + alpha1
                swarm_rho[i] = rho0 * (1 - alpha_i * (swarm_T[i] - T0))
                swarm_alpha[i] = alpha_i
                # print(swarm_rad[i],alpha_i)
        case _:
            exit("unknown geometry")

    swarm_hcond[:] = hcond0
    swarm_hcapa[:] = hcapa0
    swarm_hprod[:] = 0

    return swarm_rho, swarm_eta, swarm_hcond, swarm_hcapa, swarm_hprod, swarm_alpha, swarm_mechanism


###############################################################################


@numba.njit
def gravity_model(x, z):
    match geometry:
        case "box":
            gx = 0
            gz = -10
        case "quarter" | "half" | "eighth" | "annulus":
            g0 = 10
            gx = -x / np.sqrt(x**2 + z**2) * g0
            gz = -z / np.sqrt(x**2 + z**2) * g0
    return gx, gz


###############################################################################
