import numpy as np
import numba
import scipy
from toolbox import *
from constants import *
import global_stuff 

###################################################################################################
Router = 6370e3
Rinner = Router - 3000e3
Rmean = (Rinner + Router) / 2
top_free_slip = True
bot_free_slip = True

print(global_stuff.icase)

#icase = '1a_BA_q_32'

match global_stuff.icase:
    case '1a_BA_q_32':
      formulation = 'BA'
      geometry='quarter'
      nelz = 32
      nelx = int(2 * np.pi * Rmean / 4 / (Router - Rinner) * nelz)
    case '1a_EBA_q_32':
      formulation = 'EBA'
      geometry='quarter'
      nelz = 32
      nelx = int(2 * np.pi * Rmean / 4 / (Router - Rinner) * nelz)
    case '1a_BA_h_32':
      formulation = 'BA'
      geometry='half'
      nelz = 32
      nelx = int(2 * np.pi * Rmean / 2 / (Router - Rinner) * nelz)
    case '1a_EBA_h_32':
      formulation = 'EBA'
      geometry='half'
      nelz = 32
      nelx = int(2 * np.pi * Rmean / 2 / (Router - Rinner) * nelz)

    case _:
      exit('unknown icase value')

rho0=4000

output_folder='OUTPUT_'+global_stuff.icase

#    case "eighth":
#        nelx = int(2 * np.pi * Rmean / 8 / (Router - Rinner) * nelz)
#    case "annulus":
#        nelx = int(2 * np.pi * Rmean / 1 / (Router - Rinner) * nelz)

debug_ascii = False

solve_T = True
Tsurf = 273
Tcmb = 3273

time_scale = year
time_unit = "yr"
vel_scale = cm / year
vel_unit = "cm/yr"
p_scale = 1e6
p_unit = "MPa"

T0 = Tsurf

end_time = 5000e6 * year
every_solution = 10
every_swarm_vtu = 10
RKorder = -1

nstep = 10

eta_ref = 1e22

CFLnb = 0.75

###############################################################################


def initial_temperature(x, z, rad, theta, nn_V):

    T = np.zeros(nn_V, dtype=np.float64)

    age_surf = 250e6 * year
    age_cmb = 250e6 * year

    kappa = 1e-6
    coeff = 0.7
    Tm = Tsurf + (Tcmb - Tsurf) * coeff

    for i in range(0, nn_V):
        T[i] = initial_temperature_hsc(rad[i], Rinner, Router, Tcmb, Tsurf, age_cmb, age_surf, Tm, kappa)
        T[i] += (0.01 * Tm * np.sin(3 * theta[i])
                 + 0.02 * Tm * np.sin(7 * theta[i])
                 + 0.03 * Tm * np.sin(9 * theta[i]))

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

    match geometry:

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
        x_T,
        z_T,
        rad_T,
        theta_T,
        Nfem_T,
        nn_T,
        hull_Tnodes,
        top_Tnodes,
        bot_Tnodes,
        left_Tnodes,
        right_Tnodes,
):

    bc_fix_T = np.zeros(Nfem_T, dtype=bool)
    bc_val_T = np.zeros(Nfem_T, dtype=np.float64)

    for i in range(0, nn_T):
        if bot_Tnodes[i]:
            bc_fix_T[i] = True
            bc_val_T[i] = Tcmb
        if top_Tnodes[i]:
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

    match global_stuff.icase:
        case '1a_BA_q_32' | '1a_EBA_q_32' | '1a_BA_h_32' | '1a_EBA_h_32' :
            swarm_alpha[:] = 3e-5
            swarm_eta[:]   = 9.72e+24
            swarm_rho[:]   = rho0 * (1 - swarm_alpha[:] * (swarm_T[:] - T0))
            swarm_hcond[:] = 5
            swarm_hcapa[:] = 1250
            swarm_hprod[:] = 0
            swarm_mechanism[:] = 1
        case '1b_BA_q_32' | '1b_EBA_q_32' | '1b_BA_h_32' | '1b_EBA_h_32' :
            swarm_alpha[:] = 3e-5
            swarm_eta[:]   = 9.72e+23
            swarm_rho[:]   = rho0 * (1 - swarm_alpha[:] * (swarm_T[:] - T0))
            swarm_hcond[:] = 5
            swarm_hcapa[:] = 1250
            swarm_hprod[:] = 0
            swarm_mechanism[:] = 1
        case _:
            raise ValueError('unknown icase value')

    return swarm_rho, swarm_eta, swarm_hcond, swarm_hcapa, swarm_hprod, swarm_alpha, swarm_mechanism


###############################################################################


@numba.njit
def gravity_model(x, z):
    g0 = 10
    gx = -x / np.sqrt(x**2 + z**2) * g0
    gz = -z / np.sqrt(x**2 + z**2) * g0
    return gx, gz


###############################################################################
