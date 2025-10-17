import numpy as np

cm=0.01
year=365.25*3600*24


Lx=11600e3
Ly=2900e3
gy=-9.81
eta_ref=1e22
solve_T=True
pressure_normalisation='surface'
p_scale=1e6 ; p_unit="MPa"
vel_scale=cm/year ; vel_unit='cm/yr'
time_scale=year ; time_unit='yr'
every_Nu=100000
TKelvin=273.15
end_time=1e9*year
Tbottom=3000+TKelvin
Ttop=0+TKelvin
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

rho0=3300
alphaT=2e-5
T0=TKelvin
hcond0=5
hcapa0=1250
eta0=2e22

print(hcond0/hcapa0/rho0 )
print( (Tbottom-Ttop)*rho0*abs(gy)*alphaT*Ly**3 / eta0 / (hcond0/hcapa0/rho0))

nelx=128
nely=32
nstep=100
CFLnb=0.5         

###############################################################################

def initial_temperature(x,y,nn_V):

    T=np.zeros(nn_V,dtype=np.float64)

    for i in range(0,nn_V):
        #T[i]=(Tbottom-Ttop)*(Ly-y_V[i])/Ly+Ttop\
        T[i]=(Tbottom+Ttop)/2\
             +11*np.cos(3.33*np.pi*x[i]/Lx)*np.sin(1*np.pi*y[i]/Ly)\
             +12*np.cos(5.55*np.pi*x[i]/Lx)*np.sin(3*np.pi*y[i]/Ly)**2\
             +13*np.cos(7.77*np.pi*x[i]/Lx)*np.sin(5*np.pi*y[i]/Ly)**3

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

    swarm_rho[:]=rho0*(1-alphaT*(swarm_T[:]-T0))
    swarm_eta[:]=eta0
    swarm_hcond[:]=hcond0
    swarm_hcapa[:]=hcapa0

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###############################################################################

