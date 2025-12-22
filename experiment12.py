import numpy as np
from constants import *

###################################################################################################

axisymmetric=True
solve_Stokes=False

geometry='half'

nelz=10

if geometry=='half': nelx=int(6.7*nelz)

eta_mantle=1e21
rho_mantle=3300

Rinner=3480e3
Router=6370e3

nstep=1
eta_ref=1e21
p_scale=1e6 ; p_unit="MPa"
vel_scale=cm/year ; vel_unit='cm/yr'
time_scale=year ; time_unit='yr'
every_solution_vtu=1
every_swarm_vtu=1
debug_ascii=True

gravity_npts=250
gravity_height=200e3
gravity_rho_ref=0

###################################################################################################

def assign_boundary_conditions_V(x_V,z_V,rad_V,theta_V,ndof_V,Nfem_V,nn_V,\
                                 hull_nodes,top_nodes,bot_nodes,left_nodes,right_nodes):

    bc_fix_V=np.zeros(Nfem_V,dtype=bool) # boundary condition, yes/no
    bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

    return bc_fix_V,bc_val_V

###################################################################################################

def particle_layout(nparticle,swarm_x,swarm_z,swarm_rad,swarm_theta,Lx,Lz):

    swarm_mat=np.zeros(nparticle,dtype=np.int32)
    swarm_mat[:]=1

    return swarm_mat

###################################################################################################

def material_model(nparticle,swarm_mat,swarm_x,swarm_z,swarm_rad,swarm_theta,swarm_exx,swarm_ezz,swarm_exz,swarm_T,swarm_p):

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)
    swarm_hcond=0
    swarm_hcapa=0
    swarm_hprod=0

    swarm_eta[:]=eta_mantle 
    swarm_rho[:]=rho_mantle

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###################################################################################################

def gravity_model(x,z):
    g0=10
    gx=-x/np.sqrt(x**2+z**2)*g0
    gz=-z/np.sqrt(x**2+z**2)*g0
    return gx,gz

###################################################################################################
