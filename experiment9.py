import numpy as np
from constants import *

###################################################################################################
axisymmetric=True

#geometry='quarter'
geometry='half'

Lx=1
Lz=1

nelz=16

if geometry=='quarter': nelx=int(3*nelz)
if geometry=='half': nelx=int(5*nelz)

Rinner=1835e3
Router=3396e3

eta_blob=1e22
rho_blob=3470
depth_blob=800e3
thickness_blob=200e3

eta_crust=1e23
rho_crust=3050
depth_crust=60e3

eta_lithosphere=6e20
rho_lithosphere=3550
depth_lithosphere=400e3

eta_uppermantle=1e21
rho_uppermantle=3575
depth_uppermantle=896e3

eta_lowermantle=2e21
rho_lowermantle=3600

rho_DT_top=0
rho_DT_bot=10000

nstep=1
eta_ref=1e21
p_scale=1e6 ; p_unit="MPa"
vel_scale=cm/year ; vel_unit='cm/yr'
time_scale=year ; time_unit='yr'
every_solution_vtu=1
every_swarm_vtu=1
nparticle_per_dim=6
averaging='geometric'
debug_ascii=False
CFLnb=0.1
end_time=100e6*year

top_free_slip=True
bot_free_slip=True

gravity_npts=250
gravity_height=200e3
gravity_rho_ref=0

###################################################################################################

def assign_boundary_conditions_V(x_V,z_V,rad_V,theta_V,ndof_V,Nfem_V,nn_V,\
                                 hull_nodes,top_nodes,bot_nodes,left_nodes,right_nodes):

    bc_fix_V=np.zeros(Nfem_V,dtype=bool) # boundary condition, yes/no
    bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

    if geometry=='quarter':
       for i in range(0,nn_V):
           if x_V[i]/Rinner<eps:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
           if z_V[i]/Rinner<eps:
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           if bot_nodes[i] and right_nodes[i]:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           if top_nodes[i] and right_nodes[i]:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           if bot_nodes[i] and left_nodes[i]:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           if top_nodes[i] and left_nodes[i]:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.


    if geometry=='half':
       for i in range(0,nn_V):
           if x_V[i]/Rinner<eps:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
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

###################################################################################################

def particle_layout(nparticle,swarm_x,swarm_z,swarm_rad,swarm_theta,Lx,Lz):

    swarm_mat=np.zeros(nparticle,dtype=np.int32)

    for ip in range(nparticle):
        if swarm_rad[ip]>Router-depth_crust :
           swarm_mat[ip]=1
        elif swarm_rad[ip]>Router-depth_lithosphere:
           swarm_mat[ip]=2
        elif swarm_rad[ip]>Router-depth_uppermantle:
           swarm_mat[ip]=3
        else:
           swarm_mat[ip]=4

        if abs(swarm_theta[ip]-np.pi/2)<np.pi/10 and \
           abs(swarm_rad[ip]-(Router-depth_blob))<thickness_blob/2:
           swarm_mat[ip]=5

    return swarm_mat

###################################################################################################

def material_model(nparticle,swarm_mat,swarm_x,swarm_z,swarm_rad,swarm_theta,\
                   swarm_exx,swarm_ezz,swarm_exz,swarm_T,swarm_p):

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)
    swarm_hcond=0
    swarm_hcapa=0
    swarm_hprod=0

    mask=(swarm_mat==1) ; swarm_eta[mask]=eta_crust       ; swarm_rho[mask]=rho_crust
    mask=(swarm_mat==2) ; swarm_eta[mask]=eta_lithosphere ; swarm_rho[mask]=rho_lithosphere
    mask=(swarm_mat==3) ; swarm_eta[mask]=eta_uppermantle ; swarm_rho[mask]=rho_uppermantle
    mask=(swarm_mat==4) ; swarm_eta[mask]=eta_lowermantle ; swarm_rho[mask]=rho_lowermantle
    mask=(swarm_mat==5) ; swarm_eta[mask]=eta_blob        ; swarm_rho[mask]=rho_blob

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###################################################################################################

def gravity_model(x,z):
    g0=3.72 #https://en.wikipedia.org/wiki/Mars
    gx=-x/np.sqrt(x**2+z**2)*g0
    gz=-z/np.sqrt(x**2+z**2)*g0
    return gx,gz

###################################################################################################
