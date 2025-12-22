import numpy as np
from constants import *

###################################################################################################

nelx=205
nelz=93

Lx=400*km
Lz=180*km

nstep=1
end_time=40e6*year
eta_ref=1e21
every_solution_vtu=1
every_swarm_vtu=1
averaging='harmonic'
#particle_distribution=1 # 0: random, 1: reg, 2: Poisson Disc, 3: pseudo-random
#nodal_projection_type=4

#debug_ascii=True
#remove_rho_profile=True

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
    swarm_mat[:]=5

    for ip in range(0,nparticle):
        if swarm_x[ip]<=300e3 and swarm_z[ip]<Lz-60e3: swarm_mat[ip]=1 # asthenosphere left
        if swarm_x[ip]>=300e3 and swarm_z[ip]<Lz-18e3: swarm_mat[ip]=2 # asthenosphere right
        if swarm_x[ip]<=300e3 and swarm_z[ip]>Lz-60e3 and swarm_z[ip]<=Lz-10e3: swarm_mat[ip]=3 # lithosphere left
        if swarm_x[ip]>=300e3 and swarm_z[ip]>Lz-18e3 and swarm_z[ip]<=Lz-8e3 : swarm_mat[ip]=4 # lithosphere right

    return swarm_mat

###################################################################################################

def material_model(nparticle,swarm_mat,swarm_x,swarm_z,swarm_rad,swarm_theta,\
                   swarm_exx,swarm_ezz,swarm_exz,swarm_T,swarm_p):

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)
    swarm_hcond=0
    swarm_hcapa=0
    swarm_hprod=0

    mask=(swarm_mat==1) ; swarm_eta[mask]=1e21 ; swarm_rho[mask]=3200 #asthenosphere left
    mask=(swarm_mat==2) ; swarm_eta[mask]=1e21 ; swarm_rho[mask]=3200 #asthenosphere right 
    mask=(swarm_mat==3) ; swarm_eta[mask]=1e22 ; swarm_rho[mask]=3300 #lithosphere left
    mask=(swarm_mat==4) ; swarm_eta[mask]=1e22 ; swarm_rho[mask]=3300 #lithosphere right
    mask=(swarm_mat==5) ; swarm_eta[mask]=1e18 ; swarm_rho[mask]=1030 #water

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###################################################################################################

def gravity_model(x,z):
    return 0.,-9.81

###################################################################################################



