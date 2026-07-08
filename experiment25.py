import numpy as np
from constants import *

CFLnb=0.75

nelx = 16
nelz = nelx

#formulation = "BA"
formulation = "EBA"

solve_T = True


#mine old
#Lx = 3000e3
#Lz = 3000e3
#Ttop = 500 + 273
#Tbottom = 3500 +273 
#alphaT = 1e-5  # thermal expansion coefficient
#hcond = 3  # thermal conductivity
#hcapa = 1200  # heat capacity
#rho0 = 2500 # reference density
#g0 = 10 # gravitational acceleration
#eta0 = 2.025e24 # Ra=1e4
#eta0 = 2.025e23 # Ra=1e5
#eta0 = 2.025e22  # Ra=1e6

# lee_13 ----------------------
Lx = 1000e3
Lz = 1000e3
g0 = 10 # gravitational acceleration
Ttop = 0 + 273
Tbottom = 3000 +273 
alphaT = 3.125e-5  # thermal expansion coefficient
rho0 = 4000 # reference density
hcapa = 1250  # heat capacity
kappa=1e-6
hcond=kappa*rho0*hcapa
eta0 = 3.750e23 # Ra=1e4
#------------------------------


end_time = 1e10 * year
every_solution_vtu = 100
every_swarm_vtu = 100
RKorder = -1
nstep = 25000

Di = alphaT * g0 * Lz / hcapa
kappa = hcond / rho0 / hcapa
Ra = alphaT * rho0 * g0 * (Lz**3) * (Tbottom - Ttop) / kappa / eta0

print("     -> Di=", Di)
print("     -> kappa=", kappa)
print("     -> Ra=", Ra)

reftime = rho0 * hcapa * Lz**2 / hcond
refvel = Lz / reftime
refTemp = Tbottom - Ttop
refPress = eta0 * hcond / rho0 / hcapa / Lz**2

print("     -> reftime %e s | %e yr" % (reftime, reftime / year))
print("     -> refvel %e m/s | %e cm/yr" % (refvel, refvel / cm * year))
print("     -> refPress %e " % refPress)

eta_ref = eta0
time_scale = year
time_unit = "yr"
vel_scale = cm / year
vel_unit = "cm/yr"
p_scale = 1e6
p_unit = "MPa"

###################################################################################################


def initial_temperature(x, z, rad, theta, nn_T):

    T = np.zeros(nn_T, dtype=np.float64)

    for i in range(0, nn_T):
        T[i] = (Tbottom - Ttop) * (Lz - z[i]) / Lz + Ttop + 10 * np.cos(np.pi * x[i] / Lx) * np.sin(np.pi * z[i] / Lz)

    return T


###################################################################################################
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

    eps = 1e-8

    bc_fix_T = np.zeros(Nfem_T, dtype=bool)
    bc_val_T = np.zeros(Nfem_T, dtype=np.float64)

    for i in range(0, nn_T):
        if z_T[i] / Lz < eps:
            bc_fix_T[i] = True
            bc_val_T[i] = Tbottom
        if z_T[i] / Lz > (1 - eps):
            bc_fix_T[i] = True
            bc_val_T[i] = Ttop

    return bc_fix_T, bc_val_T


###################################################################################################


def particle_layout(nparticle, nmat, swarm_x, swarm_z, swarm_rad, swarm_theta, Lx, Lz):

    swarm_wf = np.zeros((nmat, nparticle), dtype=np.float64)
    swarm_wf[:, :] = 1

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

    swarm_rho[:] = rho0 * (1 - alphaT * (swarm_T[:]-Ttop))
    swarm_eta[:] = eta0
    swarm_hcond[:] = hcond
    swarm_hcapa[:] = hcapa
    swarm_hprod[:] = 0
    swarm_alpha[:] = alphaT

    return swarm_rho, swarm_eta, swarm_hcond, swarm_hcapa, swarm_hprod, swarm_alpha, swarm_mechanism


###################################################################################################


def gravity_model(x, z):

    gx = 0
    gz = -g0

    return gx, gz


###################################################################################################
