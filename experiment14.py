import numpy as np
from constants import *

###################################################################################################

nelx=100
nelz=66

Lx=1000*km
Lz=660*km

CFLnb=0.01
nstep=100
eta_ref=1e22
p_scale=1e6 ; p_unit="MPa"
vel_scale=cm/year ; vel_unit='cm/yr'
time_scale=year ; time_unit='yr'
every_solution_vtu=1
every_swarm_vtu=1
every_quadpoints_vtu=10
end_time=120e6*year
dt_max=1e4*year

icase=2

###################################################################################################

def assign_boundary_conditions_V(x_V,z_V,rad_V,theta_V,ndof_V,Nfem_V,nn_V,\
                                 hull_nodes,top_nodes,bot_nodes,left_nodes,right_nodes):

    bc_fix_V=np.zeros(Nfem_V,dtype=bool) # boundary condition, yes/no
    bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

    for i in range(0,nn_V):
        if x_V[i]/Lx<eps:
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
        if x_V[i]/Lx>(1-eps):
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
        if z_V[i]/Lz<eps:
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
        if z_V[i]/Lz>(1-eps):
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

    return bc_fix_V,bc_val_V

###################################################################################################

def particle_layout(nparticle,swarm_x,swarm_z,swarm_rad,swarm_theta,Lx,Lz):

    swarm_mat=np.zeros(nparticle,dtype=np.int32)
    swarm_mat[:]=1

    for ip in range(0,nparticle):
        if (swarm_z[ip]>660e3-80e3 and swarm_z[ip]<=660e3) or (swarm_z[ip]>660e3-(80e3+250e3) and abs(swarm_x[ip]-Lx/2)<40e3):
           swarm_mat[ip]=2

    return swarm_mat

###################################################################################################

def material_model(nparticle,swarm_mat,swarm_x,swarm_z,swarm_rad,swarm_theta,\
                   swarm_exx,swarm_ezz,swarm_exz,swarm_T,swarm_p):

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)
    swarm_hcond=0
    swarm_hcapa=0
    swarm_hprod=0

    if icase==0:
       mask=(swarm_mat==1) ; swarm_eta[mask]=1e21 ; swarm_rho[mask]=3150
       mask=(swarm_mat==2) ; swarm_eta[mask]=1e22 ; swarm_rho[mask]=3300

    if icase==1:
       mask=(swarm_mat==1) ; swarm_eta[mask]=1e21 ; swarm_rho[mask]=3150
       mask=(swarm_mat==2) ;                        swarm_rho[mask]=3300

       swarm_sr=np.sqrt(0.5*(swarm_exx**2+swarm_ezz**2)+swarm_exz**2)
       for ip in range(0,nparticle):
           if swarm_mat[ip]==2:
              sr=max(1e-30,swarm_sr[ip])
              n_pow=4
              val=(4.75e11)*sr**(1./n_pow -1.)
              val=max(val,1e19)
              val=min(val,1e25)
              swarm_eta[ip]=val

    if icase==2:
       mask=(swarm_mat==1) ; swarm_rho[mask]=3150
       mask=(swarm_mat==2) ; swarm_rho[mask]=3300

       swarm_sr=np.sqrt(0.5*(swarm_exx**2+swarm_ezz**2)+swarm_exz**2)
       for ip in range(0,nparticle):
           sr=max(1e-30,swarm_sr[ip])
           if swarm_mat[ip]==1:
              n_pow=3
              val=(4.54e10)*sr**(1./n_pow -1.)
           else: 
              n_pow=4
              val=(4.75e11)*sr**(1./n_pow -1.)
           val=max(val,1e19)
           val=min(val,1e25)
           swarm_eta[ip]=val


    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###################################################################################################

def gravity_model(x,z):
    return 0.,-10.

###################################################################################################
