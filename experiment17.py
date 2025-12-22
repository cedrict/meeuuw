import numpy as np
from constants import *

###################################################################################################

nelx=32
nelz=32

Lx=512*km
Lz=512*km

nstep=1
end_time=0
eta_ref=1e21
every_solution_vtu=1
every_swarm_vtu=1
averaging='arithmetic'
#nparticle_per_dim=4
particle_distribution=1 # 0: random, 1: reg, 2: Poisson Disc, 3: pseudo-random
nodal_projection_type=1

#debug_ascii=True
#remove_rho_profile=True

particle_rho_projection='least_squares'
particle_eta_projection='least_squares'

p_scale=1e6 ; p_unit="MPa"
vel_scale=cm/year ; vel_unit='cm/yr'
time_scale=year ; time_unit='yr'

###################################################################################################

def assign_boundary_conditions_V(x_V,z_V,rad_V,theta_V,ndof_V,Nfem_V,nn_V,\
                                 hull_nodes,top_nodes,bot_nodes,left_nodes,right_nodes):

    bc_fix_V=np.zeros(Nfem_V,dtype=bool) # boundary condition, yes/no
    bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

    for i in range(0,nn_V):
        if x_V[i]/Lx<eps:
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
        if x_V[i]/Lx>(1-eps):
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
        if z_V[i]/Lz<eps:
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
        if z_V[i]/Lz>(1-eps):
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

    return bc_fix_V,bc_val_V

###################################################################################################

def particle_layout(nparticle,swarm_x,swarm_z,swarm_rad,swarm_theta,Lx,Lz):

    swarm_mat=np.zeros(nparticle,dtype=np.int32)
    swarm_mat[:]=1 # air

    for ip in range(0,nparticle):
        if swarm_z[ip]<384e3: swarm_mat[ip]=2 # asthenosphere 

        if (swarm_x[ip]-Lx/2)**2+(swarm_z[ip]-Lz/2)**2<64e3**2 : swarm_mat[ip]=3 # sphere

    return swarm_mat

###################################################################################################

def material_model(nparticle,swarm_mat,swarm_x,swarm_z,swarm_rad,swarm_theta,\
                   swarm_exx,swarm_ezz,swarm_exz,swarm_T,swarm_p):

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)
    swarm_hcond=0
    swarm_hcapa=0
    swarm_hprod=0

    mask=(swarm_mat==1) ; swarm_eta[mask]=1e18 ; swarm_rho[mask]=1   #air
    mask=(swarm_mat==2) ; swarm_eta[mask]=1e21 ; swarm_rho[mask]=3200 #asthenosphere
    mask=(swarm_mat==3) ; swarm_eta[mask]=1e22 ; swarm_rho[mask]=3300 #sphere

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###################################################################################################

def gravity_model(x,z):
    return 0.,-9.81

###################################################################################################



