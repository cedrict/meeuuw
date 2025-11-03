import numpy as np
import numba

###############################################################################

def assign_parameters(icase):
    Ra=1e2
    match(icase):
         case 1 :
             sigma_y=1.
             gamma_y=np.log(1.)
         case 2 :
             sigma_y = 1
             gamma_y=np.log(1.)
         case 3 :
             sigma_y = 1
             gamma_y=np.log(10.)
         case 4 :
             sigma_y = 1
             gamma_y=np.log(10.)
         case 5 :
             sigma_y=4.
             gamma_y=np.log(10.)
         case _ :
             exit('pb in assign_parameters')
    return Ra,sigma_y,gamma_y

###############################################################################

Lx=1
Ly=1
eta_ref=1
solve_T=True
vel_scale=1 ; vel_unit=' '
time_scale=1 ; time_unit=' '
p_scale=1 ; p_unit=' '
Ttop=0
Tbottom=1
alphaT=1e-4
hcond=1  
hcapa=1 
rho0=1
Ra=1e4
TKelvin=0
pressure_normalisation='surface'
every_Nu=1
end_time=0.25
case_tosi=1
gamma_T=np.log(1e5)
eta_star=1e-3 
eta_ref=1e-2
Ra,sigma_y,gamma_y=assign_parameters(case_tosi)
eta_min=1e-5
eta_max=1
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
geometry='box'
rho_DT_top=0
rho_DT_bot=0
gravity_npts=0
tol_ss=-1e-8

nelx=32
nely=32
nstep=10

CFLnb=0.5

###############################################################################

@numba.njit
def viscosity(T,exx,eyy,exy,y,gamma_T,gamma_y,sigma_y,eta_star,icase):
    #-------------------
    # tosi et al, case 1
    #-------------------
    if icase==1:
       val=np.exp(-gamma_T*T)
    #-------------------
    # tosi et al, case 2
    #-------------------
    elif icase==2:
       e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
       e=max(e,1e-12)
       eta_lin=np.exp(-gamma_T*T)
       eta_plast=eta_star + sigma_y/(np.sqrt(2.)*e)
       val=2./(1./eta_lin + 1./eta_plast)
    #-------------------
    # tosi et al, case 3
    #-------------------
    elif icase==3:
       val=np.exp(-gamma_T*T+gamma_y*(1-y))
    #-------------------
    # tosi et al, case 4
    #-------------------
    elif icase==4:
       e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
       eta_lin=np.exp(-gamma_T*T+gamma_y*(1-y))
       eta_plast=eta_star + sigma_y/(np.sqrt(2)*e)
       val=2/(1/eta_lin + 1/eta_plast)
    #-------------------
    # tosi et al, case 5
    #-------------------
    elif icase==5:
       e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
       eta_lin=np.exp(-gamma_T*T+gamma_y*(1-y))
       eta_plast=eta_star + sigma_y/(np.sqrt(2)*e)
       val=2/(1/eta_lin + 1/eta_plast)
    val=min(2.0,val)
    val=max(1.e-5,val)
    return val

###############################################################################

def initial_temperature(x,y,rad,theta,nn_V):

    T=np.zeros(nn_V,dtype=np.float64)

    for i in range(0,nn_V):
        T[i]=(Tbottom-Ttop)*(Ly-y[i])/Ly+Ttop\
             +0.01*np.cos(np.pi*x[i]/Lx)*np.sin(np.pi*y[i]/Ly)

    return T

###############################################################################

###############################################################################
# free slip on all sides

def assign_boundary_conditions_V(x_V,y_V,rad_V,theta_V,ndof_V,Nfem_V,nn_V):

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

def assign_boundary_conditions_T(x_V,y_V,rad_V,theta_V,Nfem_T,nn_V):

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

def particle_layout(nparticle,swarm_x,swarm_y,swarm_rad,swarm_theta,Lx,Ly):

    swarm_mat=np.zeros(nparticle,dtype=np.int32)

    swarm_mat[:]=1

    return swarm_mat

###############################################################################

def material_model(nparticle,swarm_mat,swarm_x,swarm_y,swarm_rad,swarm_theta,swarm_exx,swarm_eyy,swarm_exy,swarm_T,swarm_p):

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)
    swarm_hcond=np.zeros(nparticle,dtype=np.float64)
    swarm_hcapa=np.zeros(nparticle,dtype=np.float64)
    swarm_hprod=np.zeros(nparticle,dtype=np.float64)

    swarm_rho[:]=rho0*(1-alphaT*swarm_T[:])

    for ip in range(0,nparticle):
        swarm_eta[ip]=viscosity(swarm_T[ip],swarm_exx[ip],swarm_eyy[ip],swarm_exy[ip],swarm_y[ip],\
                                gamma_T,gamma_y,sigma_y,eta_star,case_tosi)
    swarm_hcond[:]=1
    swarm_hcapa[:]=1
    swarm_hprod[:]=0

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###############################################################################

def gravity_model(x,y):
    gx=0
    gy=-Ra/alphaT 
    return gx,gy

###############################################################################
