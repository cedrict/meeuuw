import numpy as np
from constants import *
from prem import * 
from scipy import interpolate

#-----------------------------------------
# rho_profile:
# 0: constant value rho_mantle
# 1: PREM profile
# 2: STW105 http://ds.iris.edu/ds/products/emc-stw105/
# eta_profile:
# 0: constant value eta_mantle
# 1: Ciskova et al 2012 (A)
# 2: Ciskova et al 2012 (B)

Lx=1200e3
Ly=600e3

nelx=100
nely=50

R_blob=80e3
y_blob=350e3
eta_blob=1e20
rho_blob=3200

rho_profile=0
eta_profile=0

rho_mantle=3300 # used if rho_profile=0
eta_mantle=1e21 # used if eta_profile=0

rho_air=0
rho_core=10000

#-----------------------------------------
#------do not change these parameters ----
#-----------------------------------------
nstep=2
gy=-10
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
particle_distribution=1 # 0: random, 1: reg, 2: Poisson Disc, 3: pseudo-random
nparticle_per_dim=7
averaging='geometric'
formulation='BA'
debug_ascii=False
debug_nan=False
CFLnb=0.25
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

    #density

    if rho_profile==0:
       mask=(swarm_mat==1) ; swarm_rho[mask]=rho_mantle 
    elif rho_profile==1:
       for ip in range(0,nparticle):
           swarm_rho[ip]=prem_density(6368e3-Ly+swarm_y[ip])
    elif rho_profile==2:
       momo=np.loadtxt('DATA/rho_stw105.ascii')
       depths=momo[:,0] #; print(depths)
       rho105=momo[:,1]   #; print(etaA)
       f=interpolate.interp1d(depths,rho105)
       swarm_rho[:]=f(6368e3-Ly+swarm_y)


    else:
       exit('wrong rho_profile value')

    #viscosity

    if eta_profile==0:
       mask=(swarm_mat==1) ; swarm_eta[mask]=eta_mantle 
    elif eta_profile==1:
       momo=np.loadtxt('DATA/civs12.ascii')
       depths=momo[:,0] #; print(depths)
       etaA=momo[:,1]   #; print(etaA)
       f=interpolate.interp1d(depths,etaA)
       swarm_eta[:]=f(Ly-swarm_y)
       swarm_eta[:]=10**swarm_eta[:]
    elif eta_profile==2:
       momo=np.loadtxt('DATA/civs12.ascii')
       depths=momo[:,0] #; print(depths)
       etaB=momo[:,2]   #; print(etaB)
       f=interpolate.interp1d(depths,etaB)
       swarm_eta[:]=f(Ly-swarm_y)
       swarm_eta[:]=10**swarm_eta[:]
    elif eta_profile==3:
       for ip in range(0,nparticle):
           yip=swarm_y[ip]
           #insert here your profile
           #if yip<100e3:
           #   swarm_eta[ip]=...
           #elif ...
           #etc ...
    else:
       exit('wrong eta_profile value')

    mask=(swarm_mat==2) ; swarm_eta[mask]=eta_blob ; swarm_rho[mask]=rho_blob

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###############################################################################


