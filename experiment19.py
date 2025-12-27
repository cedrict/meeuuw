import numpy as np
from constants import *

###################################################################################################

Lx=1
Lz=1

nelx=32
nelz=32

nstep=1
end_time=0
eta_ref=1
every_solution_vtu=1
every_swarm_vtu=1
debug_ascii=True
pressure_normalisation='volume'
RKorder=-1 # particles collocated with quadrature points
compute_L2_errors=True

###################################################################################################

def assign_boundary_conditions_V(x_V,z_V,rad_V,theta_V,ndof_V,Nfem_V,nn_V,\
                                 hull_nodes,top_nodes,bot_nodes,left_nodes,right_nodes):

    bc_fix_V=np.zeros(Nfem_V,dtype=bool) # boundary condition, yes/no
    bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

    for i in range(0,nn_V):
        if x_V[i]/Lx<eps:
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
        if x_V[i]/Lx>(1-eps):
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
        if z_V[i]/Lz<eps:
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
        if z_V[i]/Lz>(1-eps):
           bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
           bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

    return bc_fix_V,bc_val_V

###################################################################################################

def particle_layout(nparticle,swarm_x,swarm_z,swarm_rad,swarm_theta,Lx,Lz):

    swarm_mat=np.ones(nparticle,dtype=np.int32)

    return swarm_mat

###################################################################################################

def material_model(nparticle,swarm_mat,swarm_x,swarm_z,swarm_rad,swarm_theta,\
                   swarm_exx,swarm_ezz,swarm_exz,swarm_T,swarm_p):

    swarm_rho=np.ones(nparticle,dtype=np.float64)
    swarm_eta=np.ones(nparticle,dtype=np.float64)
    swarm_hcond=0
    swarm_hcapa=0
    swarm_hprod=0

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###################################################################################################

def gravity_model(x,y):

    gx=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
        (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
        (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
        1.-4.*y+12.*y*y-8.*y*y*y)

    gz=((8.-48.*y+48.*y*y)*x*x*x+
        (-12.+72.*y-72.*y*y)*x*x+
        (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
        12.*y*y+24.*y*y*y-12.*y**4)

    return gx,gz

###################################################################################################
