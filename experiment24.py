import numpy as np 
import numba
import math
import scipy
from constants import *

###################################################################################################
# Murphy & King, JGR, 2024 

Lz=3389e3-1830e3
Lx=Lz
solve_T=True
Tsurf=220
deltaT = 1.2*1500
Tcmb=Tsurf+deltaT

vel_scale=cm/year ; vel_unit='cm/yr'
p_scale=1e6 ; p_unit="MPa"

alphaT=2e-5 # thermal exp
hcapa=1250  # C_p
kappa=1e-6  # heat diff
rho0=3500   
hcond=kappa*rho0*hcapa   # kappa = k / (rho0 Cp) heat conductivity

end_time=1000e6*year
every_solution_vtu=10
every_swarm_vtu=10
RKorder=-1

compute_plith=False
           
nelz=64
nelx=int(Lx/Lz*nelz)
nstep=10000

eta_ref=1e22 # purely numerical param ~ avrg viscosity

###############################################################################

def initial_temperature(x,z,rad,theta,nn_V):

    T=np.zeros(nn_V,dtype=np.float64)

    age = 100e6*year #in years, converted to seconds
    Tm = 1720 #K see table 1 

    for i in range(0,nn_V):
        if z[i] > Lz/2: #Top half
           T[i] = Tsurf + ((Tm-Tsurf) * math.erf((Lz-z[i])/(2*np.sqrt(age*kappa))))
        else:  #Bottom half
           T[i] = Tcmb - ((Tcmb-Tm) * math.erf(z[i]/(2*np.sqrt(age*kappa))))

        T[i]+=0.02*Tm*np.cos(3*np.pi*x[i]/Lx)*np.sin(5*np.pi*z[i]/Lz)\
             +0.03*Tm*np.cos(5*np.pi*x[i]/Lx)*np.sin(4*np.pi*z[i]/Lz)\
             +0.04*Tm*np.cos(7*np.pi*x[i]/Lx)*np.sin(3*np.pi*z[i]/Lz)

    return T

###############################################################################
# free slip on all sides

def assign_boundary_conditions_V(x_V,z_V,rad_V,theta_V,ndof_V,Nfem_V,nn_V,\
                                 hull_nodes,top_nodes,bot_nodes,left_nodes,right_nodes):

    bc_fix_V=np.zeros(Nfem_V,dtype=bool) # boundary condition, yes/no
    bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

    for i in range(0,nn_V):
        if x_V[i]/Lx<eps:
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
        if x_V[i]/Lx>(1-eps):
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
        if z_V[i]/Lz<eps:
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
        if z_V[i]/Lz>(1-eps):
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

    return bc_fix_V,bc_val_V

###############################################################################

def assign_boundary_conditions_T(x_V,z_V,rad_V,theta_V,Nfem_T,nn_V):

    bc_fix_T=np.zeros(Nfem_T,dtype=bool)  
    bc_val_T=np.zeros(Nfem_T,dtype=np.float64) 

    for i in range(0,nn_V):
        if z_V[i]<eps:
           bc_fix_T[i]=True ; bc_val_T[i]=Tcmb
        if z_V[i]>(Lz-eps):
           bc_fix_T[i]=True ; bc_val_T[i]=Tsurf

    return bc_fix_T,bc_val_T

###############################################################################

def particle_layout(nparticle,swarm_x,swarm_z,swarm_rad,swarm_theta,Lx,Lz):

    swarm_mat=np.zeros(nparticle,dtype=np.int32)
    swarm_mat[:]=1

    return swarm_mat

###############################################################################

def material_model(nparticle,swarm_mat,swarm_x,swarm_z,swarm_rad,swarm_theta,\
                   swarm_exx,swarm_ezz,swarm_exz,swarm_T,swarm_p):

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)
    swarm_hcond=np.zeros(nparticle,dtype=np.float64)
    swarm_hcapa=np.zeros(nparticle,dtype=np.float64)
    swarm_hprod=np.zeros(nparticle,dtype=np.float64)

    swarm_rho[:]=rho0*(1-alphaT*(swarm_T[:]-Tsurf))

    Ea = 117e3  #J/mol (Activation energy)
    Va = 6.6e-6 #m3/mol (Activation Volume)

    eta0 = 1e21 #Pa s (Reference Viscosity)

    #Add in different layers of viscosity (higher A is stronger layer)
    A=np.zeros(nparticle,dtype=np.float64)

    for i in range(nparticle):
        if Lz-swarm_z[i] > 1000e3:
           A[i] = 10
        elif Lz-swarm_z[i] > 100e3:
           A[i] = 0.1
        else:
           A[i] = 10

    #Temperature and pressure dependent eta
    for i in range(nparticle):
        a = (Ea + swarm_p[i]*Va)/(Rgas*swarm_T[i]) 
        b = (Ea + swarm_p[i]*Va)/(Rgas*(deltaT + Tsurf))                                  
        swarm_eta[i] = A[i] * eta0 * np.exp(a-b) #Pa s
        #Manual cutoffs for when viscosity is too high in the lithosphere, or too low
        swarm_eta[i]=min(swarm_eta[i],1e25)
        swarm_eta[i]=max(swarm_eta[i],1e19)

    swarm_hcond[:]=hcond
    swarm_hcapa[:]=hcapa
    swarm_hprod[:]=0

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###############################################################################

def gravity_model(x,y):
    gx=0
    gz=-3.72
    return gx,gz

###############################################################################
