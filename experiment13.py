import numpy as np
from constants import *

#-----------------------------------------

nelz=48
nelx=48

Lx=512*km
Lz=512*km

CFLnb=0.25
nstep=200
eta_ref=1e22
p_scale=1e6 ; p_unit="MPa"
vel_scale=cm/year ; vel_unit='cm/yr'
time_scale=year ; time_unit='yr'
every_solution_vtu=1
every_swarm_vtu=1
nparticle_per_dim=5
averaging='geometric'
debug_ascii=True
end_time=100e6*year

eta_mantle=1e21
rho_mantle=3200
eta_block=1e22
rho_block=3208

nsamplepoints=1
xsamplepoints=[256e3]
zsamplepoints=[384e3]

#remove_rho_profile=True

###############################################################################

def assign_boundary_conditions_V(x_V,z_V,rad_V,theta_V,ndof_V,Nfem_V,nn_V,\
                                 hull_nodes,top_nodes,bot_nodes,left_nodes,right_nodes):

    eps=1e-8

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

###############################################################################

def particle_layout(nparticle,swarm_x,swarm_z,swarm_rad,swarm_theta,Lx,Lz):

    swarm_mat=np.zeros(nparticle,dtype=np.int32)
    swarm_mat[:]=1

    for ip in range(0,nparticle):
        if abs(swarm_x[ip]-Lx/2)<64e3 and abs(swarm_z[ip]-384e3)<64e3:
           swarm_mat[ip]=2

    return swarm_mat

###############################################################################

def material_model(nparticle,swarm_mat,swarm_x,swarm_z,swarm_rad,swarm_theta,\
                   swarm_exx,swarm_ezz,swarm_exz,swarm_T,swarm_p):

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)
    swarm_hcond=0
    swarm_hcapa=0
    swarm_hprod=0

    mask=(swarm_mat==1) ; swarm_eta[mask]=eta_mantle ; swarm_rho[mask]=rho_mantle
    mask=(swarm_mat==2) ; swarm_eta[mask]=eta_block  ; swarm_rho[mask]=rho_block

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###############################################################################

def gravity_model(x,z):
    gx=0
    gz=-10
    return gx,gz

###############################################################################

