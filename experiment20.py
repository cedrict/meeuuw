import numpy as np
from constants import *

###################################################################################################

# written for review of Jiang et al Tectonoiphysics

Lx=50e3 
Lz=14e3
eta_ref=1e22
p_scale=1e6 ; p_unit="MPa"
vel_scale=cm/year ; vel_unit='cm/yr'
time_scale=year ; time_unit='yr'
every_solution_vtu=1
every_swarm_vtu=1
averaging='harmonic'
nparticle_per_dim=8

CFLnb=0

nelx=128
nelz=128
nstep=1
end_time=25e6*year

ubc=1.e-3/year

###################################################################################################

def assign_boundary_conditions_V(x_V,z_V,rad_V,theta_V,ndof_V,Nfem_V,nn_V,\
                                 hull_nodes,top_nodes,bot_nodes,left_nodes,right_nodes):

    bc_fix_V=np.zeros(Nfem_V,dtype=bool) # boundary condition, yes/no
    bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

    for i in range(0,nn_V):
        if x_V[i]/Lx<eps:
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=ubc
        if x_V[i]/Lx>(1-eps):
           if z_V[i]< 1e3:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=ubc
           elif z_V[i]< 1.3e3:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]= -(z_V[i]-1e3)/300*ubc + ubc
           else:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
        if z_V[i]/Lz<eps:
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=ubc
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

    return bc_fix_V,bc_val_V

###################################################################################################

def particle_layout(nparticle,swarm_x,swarm_z,swarm_rad,swarm_theta,Lx,Lz):

    swarm_mat=np.zeros(nparticle,dtype=np.int32)
   
    # 0: air
    # 1: sand1
    # 2: sand2
    # 3: sand3
    # 4: salt1
    # 5: sand4
    # 6: salt2
    # 7: sand5
    # 8: sand6
    # 9: sand7
    # 10: frictional layer
    # 11: rigid base

    for ip in range(0,nparticle):
        if swarm_z[ip]<1e3:
           swarm_mat[ip]=11
        elif swarm_z[ip]<1.3e3:
           swarm_mat[ip]=10
        elif swarm_z[ip]<2e3:
           swarm_mat[ip]=9
        elif swarm_z[ip]<2.6e3:
           swarm_mat[ip]=8
        elif swarm_z[ip]<3.2e3:
           swarm_mat[ip]=7
        elif swarm_z[ip]<3.7e3:
           swarm_mat[ip]=6
        elif swarm_z[ip]<4.7e3:
           swarm_mat[ip]=5
        elif swarm_z[ip]<5.2e3:
           swarm_mat[ip]=4
        elif swarm_z[ip]<6e3:
           swarm_mat[ip]=3
        elif swarm_z[ip]<7e3:
           swarm_mat[ip]=2
        elif swarm_z[ip]<8e3:
           swarm_mat[ip]=1

    return swarm_mat

###################################################################################################

def material_model(nparticle,swarm_mat,swarm_x,swarm_z,swarm_rad,swarm_theta,\
                   swarm_exx,swarm_ezz,swarm_exz,swarm_T,swarm_p):

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)
    swarm_hcond=0
    swarm_hcapa=0
    swarm_hprod=0

    mask=(swarm_mat==0)  ; swarm_eta[mask]=1e18 ; swarm_rho[mask]=1
    mask=(swarm_mat==1)  ; swarm_eta[mask]=1e23 ; swarm_rho[mask]=2500
    mask=(swarm_mat==2)  ; swarm_eta[mask]=1e23 ; swarm_rho[mask]=2500
    mask=(swarm_mat==3)  ; swarm_eta[mask]=1e23 ; swarm_rho[mask]=2600
    mask=(swarm_mat==4)  ; swarm_eta[mask]=1e18 ; swarm_rho[mask]=2200
    mask=(swarm_mat==5)  ; swarm_eta[mask]=1e23 ; swarm_rho[mask]=2650
    mask=(swarm_mat==6)  ; swarm_eta[mask]=1e18 ; swarm_rho[mask]=2200
    mask=(swarm_mat==7)  ; swarm_eta[mask]=1e23 ; swarm_rho[mask]=2700
    mask=(swarm_mat==8)  ; swarm_eta[mask]=1e23 ; swarm_rho[mask]=2700
    mask=(swarm_mat==9)  ; swarm_eta[mask]=1e23 ; swarm_rho[mask]=2700
    mask=(swarm_mat==10) ; swarm_eta[mask]=1e23 ; swarm_rho[mask]=2700
    mask=(swarm_mat==11) ; swarm_eta[mask]=1e23 ; swarm_rho[mask]=3000

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###################################################################################################

def gravity_model(x,z):
    return 0.,-9.81

###################################################################################################
