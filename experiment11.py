import numpy as np
from constants import *

###################################################################################################

solve_T = True
geometry = "eighth"

nelz = 64
nelx = int(nelz * 0.8)

alpha = 2e-5
eta_mantle = 5e22
rho0 = 3000
g0 = 9.8
TKelvin = 273.15
Rinner = 3480e3
Router = 6000e3  # 370e3
hcond0 = 4
hcapa0 = 1250

# Tadiab_surface=1250+TKelvin
# Tadiab_bottom=Tadiab_surface*np.exp(alpha*g0/Cp*(Router-Rinner))

Tsurface = 1250 + TKelvin
Tbottom = 2250 + TKelvin

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
averaging = "geometric"
debug_ascii = True
end_time = 1000e6 * year

top_free_slip = True
bot_free_slip = True

print(
    "Ra=",
    (Tbottom - Tsurface) * rho0 * g0 * alpha * (Router - Rinner) ** 3 / eta_mantle / (hcond0 / hcapa0 / rho0),
)

###################################################################################################


def assign_boundary_conditions_V(
    x_V,
    y_V,
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
        if x_V[i] / Rinner < eps:
            bc_fix_V[i * ndof_V] = True
            bc_val_V[i * ndof_V] = 0.0
        # if y_V[i]/Rinner<eps:
        #   bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
        if right_nodes[i]:
            bc_fix_V[i * ndof_V] = True
            bc_val_V[i * ndof_V] = 0.0
            bc_fix_V[i * ndof_V + 1] = True
            bc_val_V[i * ndof_V + 1] = 0.0

        if bot_nodes[i] and left_nodes[i]:
            bc_fix_V[i * ndof_V] = True
            bc_val_V[i * ndof_V] = 0.0
            bc_fix_V[i * ndof_V + 1] = True
            bc_val_V[i * ndof_V + 1] = 0.0
        # if bot_nodes[i] and right_nodes[i]:
        #   bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
        #   bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
        if top_nodes[i] and left_nodes[i]:
            bc_fix_V[i * ndof_V] = True
            bc_val_V[i * ndof_V] = 0.0
            bc_fix_V[i * ndof_V + 1] = True
            bc_val_V[i * ndof_V + 1] = 0.0
        # if top_nodes[i] and right_nodes[i]:
        #   bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
        #   bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

    return bc_fix_V, bc_val_V


###################################################################################################

def particle_layout(nparticle, nmat, swarm_x, swarm_z, swarm_rad, swarm_theta, Lx, Lz):

    swarm_wf = np.zeros((nmat, nparticle), dtype=np.float64)
    swarm_wf[0,:] = 1

    material_names = ["mantle"]

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
    swarm_hcond = np.zeros(nparticle, dtype=np.float64)
    swarm_hcapa = np.zeros(nparticle, dtype=np.float64)
    swarm_hprod = np.zeros(nparticle, dtype=np.float64)
    swarm_alpha = np.zeros(nparticle, dtype=np.float64)
    swarm_mechanism = np.zeros(nparticle, dtype=np.int32)

    swarm_hcond[:] = hcond0
    swarm_hcapa[:] = hcapa0
    swarm_hprod[:] = 0
    swarm_eta[:] = eta_mantle
    swarm_rho[:] = rho0 * (1 - alpha * (swarm_T[:] - Tsurface))
    swarm_alpha[:] = alpha 

    return swarm_rho, swarm_eta, swarm_hcond, swarm_hcapa, swarm_hprod, swarm_alpha, swarm_mechanism

###################################################################################################


def gravity_model(x, z):
    gx = -x / np.sqrt(x**2 + z**2) * g0
    gz = -z / np.sqrt(x**2 + z**2) * g0
    return gx, gz


###################################################################################################


def initial_temperature(x, z, rad, theta, nn_T):

    T = np.zeros(nn_T, dtype=np.float64)

    for i in range(0, nn_T):
        # T[i]=Tadiab_surface*np.exp(alpha*g0/Cp*(Router-rad[i]))
        T[i] = (Tbottom - Tsurface) / (1.0 / Router - 1.0 / Rinner) * (
            -1 / rad[i] + 0.5 * (1.0 / Rinner + 1.0 / Router)
        ) + 0.5 * (Tbottom + Tsurface)

    return T


###################################################################################################


def assign_boundary_conditions_T(x_T,z_T,rad_T,theta_T,Nfem_T,nn_T,hull_Tnodes,\
                                 top_Tnodes,bot_Tnodes,left_Tnodes,right_Tnodes):

    eps = 1e-8

    bc_fix_T = np.zeros(Nfem_T, dtype=bool)
    bc_val_T = np.zeros(Nfem_T, dtype=np.float64)

    for i in range(0, nn_T):
        if abs(rad_T[i] - Rinner) / Rinner < eps:
            bc_fix_T[i] = True
            bc_val_T[i] = Tbottom
            if theta_T[i] > np.pi * 0.45:
                bc_val_T[i] = Tbottom + 300

        if abs(rad_T[i] - Router) / Router < eps:
            bc_fix_T[i] = True
            bc_val_T[i] = Tsurface

    return bc_fix_T, bc_val_T


###################################################################################################
