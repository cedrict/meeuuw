import numpy as np
from constants import *
from prem import * 
from scipy import interpolate

#-----------------------------------------

Lx=1
Ly=1

nelx=128
nely=64

R_blob=200e3
eta_blob=1e21
rho_blob=3200

eta_mantle=1e21
rho_mantle=3300

Rinner=3480e3
Router=6370e3

rho_DT_top=0
rho_DT_bot=10000

nstep=10
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
nparticle_per_dim=10
averaging='geometric'
formulation='BA'
debug_ascii=True
debug_nan=False
CFLnb=0.5
end_time=100e6*year
tol_ss=-1e-8

geometry='quarter'

gravity_npts=250
gravity_height=200e3
gravity_rho_ref=0

#blob='ball'
blob='half_ball'
#blob='banaan'

###############################################################################

def assign_boundary_conditions_V(x_V,y_V,rad_V,theta_V,ndof_V,Nfem_V,nn_V):

    eps=1e-8

    bc_fix_V=np.zeros(Nfem_V,dtype=bool) # boundary condition, yes/no
    bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

    if geometry=='quarter':
       for i in range(0,nn_V):
           if abs(rad_V[i]-Rinner)<eps:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           if abs(rad_V[i]-Router)<eps:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           if x_V[i]<eps:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
              #bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           if y_V[i]/Ly<eps:
              #bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

    return bc_fix_V,bc_val_V

###############################################################################

def particle_layout(nparticle,swarm_x,swarm_y,swarm_rad,swarm_theta,Lx,Ly):

    swarm_mat=np.zeros(nparticle,dtype=np.int32)

    if blob=='ball':
       x_blob=3500e3
       y_blob=3500e3
       for ip in range(nparticle):
           if (swarm_x[ip]-x_blob)**2+(swarm_y[ip]-y_blob)**2<R_blob**2:
              swarm_mat[ip]=2
           else:
              swarm_mat[ip]=1

    if blob=='banaan':
       for ip in range(nparticle):
           if abs(swarm_theta[ip]-np.pi/4)<np.pi/32 and \
              abs(swarm_rad[ip]-(Rinner+Router)/2)<(Router-Rinner)/16:
              swarm_mat[ip]=2
           else:
              swarm_mat[ip]=1

    if blob=='half_ball':
       x_blob=0
       y_blob=5000e3
       for ip in range(nparticle):
           if (swarm_x[ip]-x_blob)**2+(swarm_y[ip]-y_blob)**2<R_blob**2:
              swarm_mat[ip]=2
           else:
              swarm_mat[ip]=1

    return swarm_mat

###############################################################################

def material_model(nparticle,swarm_mat,swarm_x,swarm_y,swarm_rad,swarm_theta,swarm_exx,swarm_eyy,swarm_exy,swarm_T):

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)
    swarm_hcond=0
    swarm_hcapa=0
    swarm_hprod=0

    mask=(swarm_mat==1) ; swarm_eta[mask]=eta_mantle ; swarm_rho[mask]=rho_mantle
    mask=(swarm_mat==2) ; swarm_eta[mask]=eta_blob ; swarm_rho[mask]=rho_blob

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###############################################################################

def gravity_model(x,y):
    g0=10
    gx=-x/np.sqrt(x**2+y**2)*g0
    gy=-y/np.sqrt(x**2+y**2)*g0
    return gx,gy

###############################################################################



