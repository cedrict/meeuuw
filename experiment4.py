import numpy as np
from constants import *

nely=32

geometry='box'

if geometry=='box':
   Lx=11600e3
   Ly=2900e3
   nelx=int(Lx/Ly*nely)
if geometry=='quarter':
   Rinner=3480e3
   Router=6370e3
   nelx=int(np.pi/4*nely*(Rinner+Router)/(Router-Rinner))
   top_free_slip=True
   bot_free_slip=True

eta_ref=1e22
solve_T=True
p_scale=1e6 ; p_unit="MPa"
vel_scale=cm/year ; vel_unit='cm/yr'
time_scale=year ; time_unit='yr'
TKelvin=273.15
end_time=1e9*year
Tbottom=3000+TKelvin
Ttop=0+TKelvin
every_solution_vtu=10
every_swarm_vtu=100
averaging='geometric'
debug_ascii=False
nparticle_per_dim=5

rho0=3300
alphaT=2e-5
T0=TKelvin
hcond0=5
hcapa0=1250
eta0=2e22

cohesion=4e6
phi=20*np.pi/180
cosphi=np.cos(phi)
sinphi=np.sin(phi)
eta_min=1e20
eta_max=6e24

print('kappa=',hcond0/hcapa0/rho0 )

if geometry=='box':
   print('Ra=', (Tbottom-Ttop)*rho0*9.81*alphaT*Ly**3 / eta0 / (hcond0/hcapa0/rho0))
if geometry=='quarter':
   print('Ra=', (Tbottom-Ttop)*rho0*9.81*alphaT*(Router-Rinner)**3 / eta0 / (hcond0/hcapa0/rho0))

nstep=1000
CFLnb=0.2         

###############################################################################

def initial_temperature(x,y,rad,theta,nn_V):

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

def assign_boundary_conditions_V(x_V,y_V,rad_V,theta_V,ndof_V,Nfem_V,nn_V,\
                                 hull_nodes,top_nodes,bot_nodes,left_nodes,right_nodes):

    eps=1e-8

    bc_fix_V=np.zeros(Nfem_V,dtype=bool) # boundary condition, yes/no
    bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

    if geometry=='box': # free slip on all sides
       for i in range(0,nn_V):
           if x_V[i]/Lx<eps:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
           if x_V[i]/Lx>(1-eps):
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
           if y_V[i]/Ly<eps:
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           if y_V[i]/Ly>(1-eps):
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

    if geometry=='quarter': # free slip not available on top/bottom
       for i in range(0,nn_V):
           #if abs(rad_V[i]-Rinner)/Rinner<eps:
           #   bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
           #   bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           #if abs(rad_V[i]-Router)/Router<eps:
           #   bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
           #   bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           if x_V[i]/Rinner<eps:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
           if y_V[i]/Rinner<eps:
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

           #no slip on all 4 corners
           if bot_nodes[i] and left_nodes[i]:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           if bot_nodes[i] and right_nodes[i]:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           if top_nodes[i] and left_nodes[i]:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           if top_nodes[i] and right_nodes[i]:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

    return bc_fix_V,bc_val_V

###############################################################################

def assign_boundary_conditions_T(x_V,y_V,rad_V,theta_V,Nfem_T,nn_V):

    eps=1e-8

    bc_fix_T=np.zeros(Nfem_T,dtype=bool)  
    bc_val_T=np.zeros(Nfem_T,dtype=np.float64) 

    if geometry=='box': # free slip on all sides
       for i in range(0,nn_V):
           if y_V[i]<eps:
              bc_fix_T[i]=True ; bc_val_T[i]=Tbottom
           if y_V[i]>(Ly-eps):
              bc_fix_T[i]=True ; bc_val_T[i]=Ttop

    if geometry=='quarter': # free slip not available on top/bottom
       for i in range(0,nn_V):
           if abs(rad_V[i]-Rinner)/Rinner<eps:
              bc_fix_T[i]=True ; bc_val_T[i]=Tbottom
           if abs(rad_V[i]-Router)/Router<eps:
              bc_fix_T[i]=True ; bc_val_T[i]=Ttop

    return bc_fix_T,bc_val_T

###################################################################################################

def particle_layout(nparticle,swarm_x,swarm_y,swarm_rad,swarm_theta,Lx,Ly):

    swarm_mat=np.zeros(nparticle,dtype=np.int32)

    swarm_mat[:]=1

    return swarm_mat

###################################################################################################

def eta_diffusion_creep(p,T):
    A=1e-18
    Q=150e3
    V=1.234e-6
    eta_df=0.5/A*np.exp((Q+p*V)/(Rgas*T))
    return eta_df

def material_model(nparticle,swarm_mat,swarm_x,swarm_y,swarm_rad,swarm_theta,swarm_exx,swarm_eyy,swarm_exy,swarm_T,swarm_p):

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)
    swarm_hcond=np.zeros(nparticle,dtype=np.float64)
    swarm_hcapa=np.zeros(nparticle,dtype=np.float64)
    swarm_hprod=np.zeros(nparticle,dtype=np.float64)

    swarm_rho[:]=rho0*(1-alphaT*(swarm_T[:]-T0))
    swarm_hcond[:]=hcond0
    swarm_hcapa[:]=hcapa0

    for ip in range(0,nparticle):
        eta_df=eta_diffusion_creep(swarm_p[ip],swarm_T[ip])
        eta_ds=1e50
        eta_v=(1./eta_df+1./eta_ds)**-1
        e=np.sqrt(0.5*(swarm_exx[ip]**2+swarm_eyy[ip]**2)+swarm_exy[ip]**2)+1e-20
        eta_pl=0.5*(swarm_p[ip]*sinphi+cohesion*cosphi)/e
        eta_eff=min(eta_v,eta_pl)
        eta_eff=max(eta_eff,eta_min)
        eta_eff=min(eta_eff,eta_max)
        swarm_eta[ip]=eta_eff
        swarm_eta[ip]=1e22

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###################################################################################################

def gravity_model(x,y):
    if geometry=='box':
       gx=0
       gy=-9.81
    if geometry=='quarter':
       g0=9.81
       gx=-x/np.sqrt(x**2+y**2)*g0
       gy=-y/np.sqrt(x**2+y**2)*g0
    return gx,gy

###################################################################################################

