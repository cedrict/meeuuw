import numpy as np 
import numba
import scipy
from constants import *

###################################################################################################
# Coltice et al , Science advances, 2019 (aso Arnould et al, G3, 2018)

# TODO: depth dependent thermal expansion
# TODO: plasticity
# TODO: initial temperature adiab profile

Lx=4500e3
Lz=3000e3
solve_T=True
Ttop=273
Tbottom=2390

vel_scale=cm/year ; vel_unit='cm/yr'
p_scale=1e6 ; p_unit="MPa"

alphaT=3e-5
T0=Ttop
hcond0=3.15   
hcapa0=716
rho0=4400

end_time=1000e6*year
every_solution_vtu=10
every_swarm_vtu=1000
RKorder=-1
           
nelz=64
nelx=int(Lx/Lz*nelz)
nstep=10000

eta_ref=1e22

CFLnb=0.75

###############################################################################

def initial_temperature(x,z,rad,theta,nn_V):

    T=np.zeros(nn_V,dtype=np.float64)

    age=250e6*year

    Tavrg=(Tbottom+Ttop)/2

    for i in range(0,nn_V):
        if z[i]<Lz/2:
           T[i]=Tbottom+(Tavrg-Tbottom)*scipy.special.erf(z[i]/2/np.sqrt(age*1e-6))
        else:
           T[i]=Ttop+(Tavrg-Ttop)*scipy.special.erf((Lz-z[i])/2/np.sqrt(age*1e-6))

        T[i]+=0.010*Tavrg*np.cos(3*np.pi*x[i]/Lx)*np.sin(5*np.pi*z[i]/Lz)\
             +0.015*Tavrg*np.cos(4*np.pi*x[i]/Lx)*np.sin(4*np.pi*z[i]/Lz)\
             +0.005*Tavrg*np.cos(5*np.pi*x[i]/Lx)*np.sin(3*np.pi*z[i]/Lz)

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
           bc_fix_T[i]=True ; bc_val_T[i]=Tbottom
        if z_V[i]>(Lz-eps):
           bc_fix_T[i]=True ; bc_val_T[i]=Ttop

    return bc_fix_T,bc_val_T

###############################################################################

def particle_layout(nparticle,swarm_x,swarm_z,swarm_rad,swarm_theta,Lx,Lz):

    swarm_mat=np.zeros(nparticle,dtype=np.int32)
    swarm_mat[:]=1

    return swarm_mat

###############################################################################
# Coltice private communication: "je me suis rendu compte que j'ai fait une erreur
# dans le redimensionnement du volume d'activation dans le modèle de 2019. Dans le 
# sup. mat. tu trouveras 13.8cm3/mol alors qu'en refaisant les calculs on trouve 
# en réalité 0.7cm3/mol. Dans une exponentielle, ça fait un petit paquet... 
# Donc si tu tentes de refaire la simu, fais gaffe à ça"
# Note that Arnould et al use the lithostatic pressure in the viscosity

def material_model(nparticle,swarm_mat,swarm_x,swarm_z,swarm_rad,swarm_theta,\
                   swarm_exx,swarm_ezz,swarm_exz,swarm_T,swarm_p):

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)
    swarm_hcond=np.zeros(nparticle,dtype=np.float64)
    swarm_hcapa=np.zeros(nparticle,dtype=np.float64)
    swarm_hprod=np.zeros(nparticle,dtype=np.float64)

    swarm_rho[:]=rho0*(1-alphaT*(swarm_T[:]-T0))

    djump=100e3/2
    d0=Lz-660e3
    E_a=160e3
    V_a=0.001e-6 # 13.8e-6
    T_0=1530
    for i in range(nparticle):
        swarm_eta[i]=1e21*np.exp(np.log(30)*(1-0.5*(1-np.tanh((d0-swarm_z[i])/djump)))) \
                    * np.exp((E_a + swarm_p[i]*V_a)/Rgas/swarm_T[i] - E_a/Rgas/T_0)
        swarm_eta[i]=min(swarm_eta[i],1e25)
        swarm_eta[i]=max(swarm_eta[i],1e20)

    swarm_hcond[:]=hcond0
    swarm_hcapa[:]=hcapa0
    swarm_hprod[:]=0

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###############################################################################

def gravity_model(x,y):
    gx=0
    gz=-10
    return gx,gz

###############################################################################
