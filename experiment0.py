import numpy as np

geometry='box'
Lz=1
eta_ref=1
solve_T=True
Ttop=0
Tbottom=1
alphaT=1e-2   # thermal expansion coefficient
hcond=1       # thermal conductivity
hcapa=1       # heat capacity
rho0=1
every_Nu=1
end_time=0.25
every_solution_vtu=10
every_swarm_vtu=100
RKorder=-1
nstep=100

###################################################################################################

icase='1a'
 
match(icase):
   case '1a':
      Lx=1
      Ra=1e4
      nelx=64
      nelz=64
   case '1b':
      Lx=1
      Ra=1e5
      nelx=64
      nelz=64
   case '1c':
      Lx=1
      Ra=1e6
      nelx=64
      nelz=64
   case '2a':
      Lx=1
      Ra=1e4
      nelx=32
      nelz=32
   case '2b':
      Lx=2.5
      Ra=1e4
      nelx=80
      nelz=32

###################################################################################################

def viscosity(x,z,T):
    match(icase):
       case '1a':
          eta=1
       case '1b':
          eta=1
       case '1c':
          eta=1
       case '2a':
          eta=np.exp(-6.907755279*T)
       case '2b':
          eta=np.exp(-9.704060528*T+4.158883083*(1-z))
    return eta

###################################################################################################

def initial_temperature(x,z,rad,theta,nn_V):

    T=np.zeros(nn_V,dtype=np.float64)

    if geometry=='box':
       for i in range(0,nn_V):
           T[i]=(Tbottom-Ttop)*(Lz-z[i])/Lz+Ttop\
                +0.01*np.cos(np.pi*x[i]/Lx)*np.sin(np.pi*z[i]/Lz)

    if geometry=='quarter':
       for i in range(0,nn_V):
           T[i]=(Tbottom-Ttop)*(Router-rad[i])/(Router-Rinner)+Ttop\
                +0.1*np.cos(2*theta[i])*np.sin(np.pi*(rad[i]-Rinner)/(Router-Rinner))
    return T

###################################################################################################
# free slip on all sides

def assign_boundary_conditions_V(x_V,z_V,rad_V,theta_V,ndof_V,Nfem_V,nn_V,\
                                 hull_nodes,top_nodes,bot_nodes,left_nodes,right_nodes):

    eps=1e-8

    bc_fix_V=np.zeros(Nfem_V,dtype=bool) # boundary condition, yes/no
    bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

    if geometry=='box':
       for i in range(0,nn_V):
           if x_V[i]/Lx<eps:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
           if x_V[i]/Lx>(1-eps):
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
           if z_V[i]/Lz<eps:
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           if z_V[i]/Lz>(1-eps):
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

    if geometry=='quarter':
       for i in range(0,nn_V):
           if abs(rad_V[i]-Rinner)<eps:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           if abs(rad_V[i]-Router)<eps:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           if x_V[i]<eps:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
           if z_V[i]/Lz<eps:
              bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
              bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

    return bc_fix_V,bc_val_V

###################################################################################################

def assign_boundary_conditions_T(x_V,z_V,rad_V,theta_V,Nfem_T,nn_V):
    
    eps=1e-8

    bc_fix_T=np.zeros(Nfem_T,dtype=bool)  
    bc_val_T=np.zeros(Nfem_T,dtype=np.float64) 

    if geometry=='box':
       for i in range(0,nn_V):
           if z_V[i]<eps:
              bc_fix_T[i]=True ; bc_val_T[i]=Tbottom
           if z_V[i]>(Lz-eps):
              bc_fix_T[i]=True ; bc_val_T[i]=Ttop

    if geometry=='quarter':
       for i in range(0,nn_V):
           if abs(rad_V[i]-Rinner)<eps:
              bc_fix_T[i]=True ; bc_val_T[i]=Tbottom
           if abs(rad_V[i]-Router)<eps:
              bc_fix_T[i]=True ; bc_val_T[i]=Ttop

    return bc_fix_T,bc_val_T

###################################################################################################

def particle_layout(nparticle,swarm_x,swarm_z,swarm_rad,swarm_theta,Lx,Lz):

    swarm_mat=np.zeros(nparticle,dtype=np.int32)

    swarm_mat[:]=1

    return swarm_mat

###################################################################################################

def material_model(nparticle,swarm_mat,swarm_x,swarm_z,swarm_rad,swarm_theta,\
                   swarm_exx,swarm_ezz,swarm_exz,swarm_T,swarm_p):

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)
    swarm_hcond=np.zeros(nparticle,dtype=np.float64)
    swarm_hcapa=np.zeros(nparticle,dtype=np.float64)
    swarm_hprod=np.zeros(nparticle,dtype=np.float64)

    swarm_rho[:]=rho0*(1-alphaT*swarm_T[:])

    for ip in range(0,nparticle):
        swarm_eta[ip]=viscosity(swarm_x[ip],swarm_z[ip],swarm_T[ip])

    swarm_hcond[:]=1
    swarm_hcapa[:]=1
    swarm_hprod[:]=0

    return swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod

###################################################################################################

def gravity_model(x,z):

    if geometry=='box':
       gx=0
       gz=-Ra/alphaT 

    if geometry=='quarter':
       g0=Ra/alphaT 
       gx=-x/np.sqrt(x*x+z*z)*g0
       gz=-z/np.sqrt(x*x+z*z)*g0

    return gx,gz

###################################################################################################
