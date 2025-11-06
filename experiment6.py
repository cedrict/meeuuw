import numpy as np

cm=0.01
year=365.25*3600*24

Lx=1400e3  # half domain!
Ly=900e3
eta_ref=1e21
solve_T=False
p_scale=1e6 ; p_unit="MPa"
vel_scale=cm/year ; vel_unit='cm/yr'
time_scale=year ; time_unit='yr'
pressure_normalisation='surface'
every_Nu=1000000
TKelvin=0
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
rho_DT_top=0
rho_DT_bot=0
geometry='box'
gravity_npts=0
tol_ss=-1e-8

#a: cosine perturbation
#b: plume

case='b'

if case=='a':
   nelx=140
   nely=140
   nstep=15
   CFLnb=0.25           
   end_time=2e5*year

if case=='b':
   nelx=140
   nely=140
   nstep=1500
   CFLnb=0.25           
   end_time=20e6*year

###############################################################################

def assign_boundary_conditions_V(x_V,y_V,rad_V,theta_V,ndof_V,Nfem_V,nn_V,\
                                 hull_nodes,top_nodes,bot_nodes,left_nodes,right_nodes):

    eps=1e-8

    bc_fix_V=np.zeros(Nfem_V,dtype=bool) # boundary condition, yes/no
    bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

    for i in range(0,nn_V):
        if x_V[i]/Lx<eps:
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
        if x_V[i]/Lx>(1-eps):
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
        if y_V[i]/Ly<eps:
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
        if y_V[i]/Ly>(1-eps):
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

    return bc_fix_V,bc_val_V

###############################################################################

def particle_layout(nparticle,swarm_x,swarm_y,swarm_rad,swarm_theta,Lx,Ly):

    swarm_mat=np.zeros(nparticle,dtype=np.int32)

    swarm_mat[:]=2 # mantle 

    if case=='a':

       for ip in range(0,nparticle):
           if swarm_y[ip]>600e3:
              swarm_mat[ip]=1 # lithosphere
           if swarm_y[ip]>700e3+7e3*np.cos(swarm_x[ip]/Lx*np.pi):
              swarm_mat[ip]=0 # sticky air

    if case=='b':
       for ip in range(0,nparticle):
           if swarm_y[ip]>600e3: swarm_mat[ip]=1 # lithosphere 
           if swarm_y[ip]>700e3: swarm_mat[ip]=0 # sticky air
           if (swarm_x[ip]-Lx)**2+(swarm_y[ip]-300e3)**2<50e3**2: 
              swarm_mat[ip]=3 # plume

    return swarm_mat

###############################################################################

def material_model(nparticle,swarm_mat,swarm_x,swarm_y,swarm_rad,swarm_theta,swarm_exx,swarm_eyy,swarm_exy,swarm_T,swarm_p):

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)
    swarm_hcond=0
    swarm_hcapa=0
    swarm_hprod=0

    mask=(swarm_mat==0) ; swarm_eta[mask]=1e19 ; swarm_rho[mask]=0
    mask=(swarm_mat==1) ; swarm_eta[mask]=1e23 ; swarm_rho[mask]=3300
    mask=(swarm_mat==2) ; swarm_eta[mask]=1e21 ; swarm_rho[mask]=3300
    mask=(swarm_mat==3) ; swarm_eta[mask]=1e20 ; swarm_rho[mask]=3200

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###############################################################################

def gravity_model(x,y):
    gx=0
    gy=-10 
    return gx,gy

###############################################################################

