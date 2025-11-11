import numpy as np 
import numba

# trha98b:
# The Rayleigh number is a measure of the
# ratio of buoyancy forces over viscous forces, and its value based on
# the temperature-dependent viscosity at the top is 100.
# at the surface T=0 , so eta_T=1
# assuming strainrate = very small at surface, then eta_e very large 
# and eta -> eta_T=1

Lx=4
Ly=1
solve_T=True
Ttop=0
Tbottom=1

alphaT=1e-4
T0=0
hcond0=1   
hcapa0=1  
rho0=1
Ra_surf=100
sigma_y=2

every_Nu=1
end_time=0.25
gamma_T=np.log(1e5)
eta_star=1e-5 
eta_ref=1e-2
every_solution_vtu=1
every_swarm_vtu=5
RKorder=4
averaging='geometric'
debug_ascii=False
nparticle_per_dim=5
           
nelx=128
nely=int(Ly/Lx*nelx)
nstep=10000


###############################################################################

@numba.njit
def viscosity(T,exx,eyy,exy,y):
    e=max(np.sqrt(exx**2+eyy**2+2*exy**2),1e-10)
    eta_T=np.exp(-gamma_T*T)
    eta_e=eta_star + sigma_y/e
    val=2/(1/eta_T + 1/eta_e)
    val=min(2.0,val)
    val=max(1.e-5,val)
    return val

###############################################################################

def initial_temperature(x,y,rad,theta,nn_V):

    T=np.zeros(nn_V,dtype=np.float64)

    for i in range(0,nn_V):
        T[i]=(Tbottom-Ttop)*(Ly-y[i])/Ly+Ttop\
            +0.01*np.cos(3*np.pi*x[i]/Lx)*np.sin(3*np.pi*y[i]/Ly)

    return T

###############################################################################
# free slip on all sides

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

    swarm_rho[:]=rho0*(1-alphaT*(swarm_T[:]-T0))

    for ip in range(0,nparticle):
        swarm_eta[ip]=viscosity(swarm_T[ip],swarm_exx[ip],swarm_eyy[ip],swarm_exy[ip],swarm_y[ip])

    swarm_hcond[:]=hcond0
    swarm_hcapa[:]=hcapa0
    swarm_hprod[:]=0

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###############################################################################

def gravity_model(x,y):
    gx=0
    gy=-Ra_surf/alphaT   *2 #visc 
    return gx,gy

###############################################################################
