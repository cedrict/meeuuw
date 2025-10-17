import numpy as np

Ly=1
eta_ref=1
solve_T=True
vel_scale=1 ; vel_unit=' '
time_scale=1 ; time_unit=' '
p_scale=1 ; p_unit=' '
Ttop=0
Tbottom=1
alphaT=1e-2   # thermal expansion coefficient
hcond=1       # thermal conductivity
hcapa=1       # heat capacity
rho0=1
TKelvin=0
pressure_normalisation='surface'
every_Nu=1
end_time=0.25
every_solution_vtu=1
every_swarm_vtu=5
every_quadpoints_vtu=5
RKorder=4
particle_distribution=0 # 0: random, 1: reg, 2: Poisson Disc, 3: pseudo-random
averaging='geometric'
formulation='BA'
debug_ascii=False
debug_nan=False

nparticle_per_dim=5
nstep=1000
CFLnb=0.5 

###############################################################################

icase='2b'
 
match(icase):
   case '1a':
      Lx=1
      Ra=1e4
      nelx=32
      nely=32
   case '1b':
      Lx=1
      Ra=1e5
      nelx=32
      nely=32
   case '1c':
      Lx=1
      Ra=1e6
      nelx=32
      nely=32
   case '2a':
      Lx=1
      Ra=1e4
      nelx=32
      nely=32
   case '2b':
      Lx=2.5
      Ra=1e4
      nelx=80
      nely=32

gy=-Ra/alphaT 

###############################################################################

def viscosity(x,y,T):

    match(icase):
       case '1a':
          eta=1
       case '1b':
          eta=1
       case '1c':
          eta=1
       case '2a':
          eta=np.exp(-6.907755279*T)
       case '2b':
          eta=np.exp(-9.704060528*T+4.158883083*(1-y))

    return eta

###############################################################################

def initial_temperature(x,y,nn_V):

    T=np.zeros(nn_V,dtype=np.float64)

    for i in range(0,nn_V):
        T[i]=(Tbottom-Ttop)*(Ly-y[i])/Ly+Ttop\
             +0.01*np.cos(np.pi*x[i]/Lx)*np.sin(np.pi*y[i]/Ly)

    return T

###############################################################################
# free slip on all sides

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

def assign_boundary_conditions_T(x_V,y_V,Nfem_T,nn_V):
    
    eps=1e-8

    bc_fix_T=np.zeros(Nfem_T,dtype=bool)  
    bc_val_T=np.zeros(Nfem_T,dtype=np.float64) 

    for i in range(0,nn_V):
        if y_V[i]<eps:
           bc_fix_T[i]=True ; bc_val_T[i]=Tbottom
        if y_V[i]>(Ly-eps):
           bc_fix_T[i]=True ; bc_val_T[i]=Ttop

    return bc_fix_T,bc_val_T

###############################################################################

def particle_layout(nparticle,swarm_x,swarm_y,Lx,Ly):

    swarm_mat=np.zeros(nparticle,dtype=np.int32)

    swarm_mat[:]=1

    return swarm_mat

###############################################################################

def material_model(nparticle,swarm_mat,swarm_x,swarm_y,swarm_exx,swarm_eyy,swarm_exy,swarm_T):

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)
    swarm_hcond=np.zeros(nparticle,dtype=np.float64)
    swarm_hcapa=np.zeros(nparticle,dtype=np.float64)
    swarm_hprod=np.zeros(nparticle,dtype=np.float64)

    swarm_rho[:]=rho0*(1-alphaT*swarm_T[:])

    for ip in range(0,nparticle):
        swarm_eta[ip]=viscosity(swarm_x[ip],swarm_y[ip],swarm_T[ip])

    swarm_hcond[:]=1
    swarm_hcapa[:]=1
    swarm_hprod[:]=0

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###############################################################################

