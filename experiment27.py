import numpy as np
from constants import *

Lx = 1.6
Lz = 3

eta_ref = 100
pressure_normalisation = "none"
end_time = 1000
every_solution = 1
every_swarm_vtu = 5

nmat = 3
nelx=16
nelz=30
#nelx = 32 
#nelz = 60 
#nelx = 48 
#nelz = 90 
#nelx=64
#nelz=120
nstep = 150 

h0 = 0.2  
A0 = 0.05 
lambdaa = 3.14 
C = 0.5

CFLnb=0.2

RKorder = 1
nparticle_per_dim = 10

###################################################################################################
# 0: fold
# 1: matrix
# 2: wall

def particle_layout(nparticle, nmat, swarm_x, swarm_z, swarm_rad, swarm_theta, Lx, Lz):

    swarm_wf = np.zeros((nmat, nparticle), dtype=np.float64)

    for ip in range(0, nparticle):

        if swarm_x[ip]>1.55: 
           swarm_wf[2, ip] = 1
        elif swarm_z[ip]< C+h0+A0*np.cos(2*np.pi*swarm_x[ip]/lambdaa) and \
             swarm_z[ip]> C+A0*np.cos(2*np.pi*swarm_x[ip]/lambdaa):
           swarm_wf[0, ip] = 1
        else:
           swarm_wf[1, ip] = 1

    material_names = ["fold", "matrix", "wall"]

    return swarm_wf, material_names


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
            bc_val_V[i * ndof_V] = -0.01
        if z_V[i] / Lz < eps:
            bc_fix_V[i * ndof_V + 1] = True
            bc_val_V[i * ndof_V + 1] = 0.0

    return bc_fix_V, bc_val_V


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

    #swarm_rho[swarm_active] = swarm_wf[0, :][swarm_active] * 1   + swarm_wf[1, :][swarm_active] * 1 + swarm_wf[2, :][swarm_active] * 1
    #swarm_eta[swarm_active] = swarm_wf[0, :][swarm_active] * 200 + swarm_wf[1, :][swarm_active] * 1 + swarm_wf[2, :][swarm_active] * 1e4

    for ip in range(0,nparticle):
        if swarm_active[ip]:
           swarm_rho[ip]=swarm_wf[0,ip] * 1   + swarm_wf[1,ip]* 1 + swarm_wf[2,ip] * 1
           swarm_eta[ip]=swarm_wf[0,ip] * 200 + swarm_wf[1,ip]* 1 + swarm_wf[2,ip] * 1e4
           #if swarm_rho[ip]<1e-6:
           #   print(ip,swarm_wf[:,ip])

    return swarm_rho, swarm_eta, swarm_hcond, swarm_hcapa, swarm_hprod, swarm_alpha, swarm_mechanism


###################################################################################################


def gravity_model(x, z):
    return 0.0, 0.0


###################################################################################################
