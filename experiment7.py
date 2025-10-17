import numpy as np
from constants import *

Lx=1200e3
Ly=600e3
nstep=2
nelx=100
nely=50

R_blob=50e3
y_blob=400e3

gy=-10

#do not modify
eta_ref=1e21
solve_T=False
RKorder=2
p_scale=1e6 ; p_unit="MPa"
vel_scale=cm/year ; vel_unit='cm/yr'
time_scale=year ; time_unit='yr'
pressure_normalisation='surface'
every_Nu=1000000
TKelvin=0
every_solution_vtu=1
every_swarm_vtu=1
every_quadpoints_vtu=500
particle_distribution=0 # 0: random, 1: reg, 2: Poisson Disc, 3: pseudo-random
nparticle_per_dim=7
averaging='geometric'
formulation='BA'
debug_ascii=True
debug_nan=False
CFLnb=0.5
end_time=100e6*year

###############################################################################

def assign_boundary_conditions_V(x_V,y_V,ndof_V,Nfem_V,nn_V):

    eps=1e-8

    bc_fix_V=np.zeros(Nfem_V,dtype=bool) # boundary condition, yes/no
    bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

    for i in range(0,nn_V):
        if x_V[i]/Lx<eps:
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
        if x_V[i]/Lx>(1-eps):
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
        if y_V[i]/Ly<eps:
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
        if y_V[i]/Ly>(1-eps):
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

    return bc_fix_V,bc_val_V

###############################################################################

def particle_layout(nparticle,swarm_x,swarm_y,Lx,Ly):

    swarm_mat=np.zeros(nparticle,dtype=np.int32)

    for ip in range(nparticle):
        if (swarm_x[ip]-Lx/2)**2+(swarm_y[ip]-y_blob)**2<R_blob**2:
           swarm_mat[ip]=2
        else:
           swarm_mat[ip]=1

    return swarm_mat

###############################################################################

def material_model(nparticle,swarm_mat,swarm_x,swarm_y,swarm_exx,swarm_eyy,swarm_exy,swarm_T):

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)
    swarm_hcond=0
    swarm_hcapa=0
    swarm_hprod=0

    mask=(swarm_mat==1) ; swarm_eta[mask]=1e21 ; swarm_rho[mask]=3300 # mantle
    mask=(swarm_mat==2) ; swarm_eta[mask]=1e20 ; swarm_rho[mask]=3200 # blob

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###############################################################################


