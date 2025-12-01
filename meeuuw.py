import numpy as np
import sys as sys
import numba
import random
import time as clock
import scipy.sparse as sps
from toolbox import *
from update_F import *
from constants import *
from scipy import sparse
from PoissonDisc import *
from pic_functions import *
from basis_functions import *
from build_matrix_plith import *
from build_matrix_stokes import *
from build_matrix_energy import *
from sample_solution import *
from define_mapping import * 
from compute_normals import *
from compute_strain_rate import *
from compute_nodal_heat_flux import *
from compute_nodal_pressure import *
from output_swarm_to_vtu import *
from output_solution_to_vtu import *
from output_quadpoints_to_vtu import *
from compute_gravity_at_point import *
from compute_gravity_fromDT_at_point import *
from compute_pressure_average import *
from project_nodal_field_onto_qpoints import *
from compute_nodal_pressure_gradient import *
from postprocessors import *

print("-----------------------------")
print("----------- MEEUUW ----------")
print("-----------------------------")

###############################################################################
# set lots of generic parameters to default value
###############################################################################

from set_default_parameters import *

###############################################################################
# experiment  0: Blankenbach et al, 1993 - isoviscous convection
# experiment  1: van Keken et al, JGR, 1997 - Rayleigh-Taylor experiment
# experiment  2: Schmeling et al, PEPI 2008 - Newtonian subduction
# experiment  3: Tosi et al, 2015 - visco-plastic convection
# experiment  4: Lindi MSc 
# experiment  5: Trompert & Hansen, Nature 1998 - convection w/ plate-like  
# experiment  6: Crameri et al, GJI 2012 (cosine perturbation & plume) 
# experiment  7: ESA workshop
# experiment  8: quarter - sinker
# experiment  9: axisymmetric Mars setup
# experiment 10: axisymmetric 4D dyn Earth benchmark of Stokes sphere
# experiment 11: rising plume 
# experiment 12: hollow earth gravity benchmark 
# experiment 13: sinking block benchmark
# experiment 14: slab detachment (Schmalholz 2011)
###############################################################################

experiment=14

if int(len(sys.argv)==8):
   experiment = int(sys.argv[1])

match(experiment):
     case 0 : from experiment0 import *
     case 1 : from experiment1 import *
     case 2 : from experiment2 import *
     case 3 : from experiment3 import *
     case 4 : from experiment4 import *
     case 5 : from experiment5 import *
     case 6 : from experiment6 import *
     case 7 : from experiment7 import *
     case 8 : from experiment8 import *
     case 9 : from experiment9 import *
     case 10: from experiment10 import *
     case 11: from experiment11 import *
     case 12: from experiment12 import *
     case 13: from experiment13 import *
     case 14: from experiment14 import *
     case _ : exit('setup - unknown experiment')  

if int(len(sys.argv)==8): # override these parameters
   nelx  = int(sys.argv[2])
   nelz  = int(sys.argv[3])
   nstep = int(sys.argv[4])
   axisymmetric=(int(sys.argv[5])==1)
   remove_rho_profile=(int(sys.argv[6])==1)
   straighten_edges=(int(sys.argv[7])==1)

###############################################################################

if geometry=='quarter' or geometry=='half' or geometry=='eighth':
   Lx=1 ; Lz=1 

ndim=2                     # number of dimensions
ndof_V=2                   # number of velocity dofs per node
nel=nelx*nelz              # total number of elements
nn_V=(2*nelx+1)*(2*nelz+1) # number of V nodes
nn_P=(nelx+1)*(nelz+1)     # number of P nodes

m_V=9 # number of velocity nodes per element
m_P=4 # number of pressure nodes per element
m_T=9 # number of temperature nodes per element

r_V=[-1, 1,1,-1, 0,1,0,-1,0]
t_V=[-1,-1,1, 1,-1,0,1, 0,0]

ndof_V_el=m_V*ndof_V

Nfem_V=nn_V*ndof_V # number of velocity dofs
Nfem_P=nn_P        # number of pressure dofs
Nfem_T=nn_V        # number of temperature dofs
Nfem=Nfem_V+Nfem_P # total nb of dofs

hx=Lx/nelx # element size in x direction
hz=Lz/nelz # element size in y direction

if geometry=='eighth': 
   opening_angle=np.pi/8
   theta_min=np.pi/4

if geometry=='quarter': 
   opening_angle=np.pi/2
   theta_min=0

if geometry=='half': 
   opening_angle=np.pi
   theta_min=-np.pi/2

if geometry=='quarter' or geometry=='half' or geometry=='eighth':
   hrad=(Router-Rinner)/nelz
   htheta=opening_angle/nelx

nparticle_per_element=nparticle_per_dim**2
nparticle=nel*nparticle_per_element

timings=np.zeros(29+1)
timings_mem=np.zeros(29+1)

#if geometry=='box': L_ref=(Lx+Lz)/2
if geometry=='box': L_ref=(hx+hz)/2
if geometry=='quarter': L_ref=(Rinner+Router)/2
if geometry=='half': L_ref=(Rinner+Router)/2
if geometry=='eighth': L_ref=(Rinner+Router)/2

###############################################################################
#@@ quadrature rule points and weights
###############################################################################

match nqperdim:
 case 3 :
  qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
  qweights=[5./9.,8./9.,5./9.]
 case 4 :
  qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
  qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
  qw4a=(18-np.sqrt(30.))/36.
  qw4b=(18+np.sqrt(30.))/36.
  qcoords=[-qc4a,-qc4b,qc4b,qc4a]
  qweights=[qw4a,qw4b,qw4b,qw4a]
 case 5 :
  qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
  qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
  qc5c=0.
  qw5a=(322.-13.*np.sqrt(70.))/900.
  qw5b=(322.+13.*np.sqrt(70.))/900.
  qw5c=128./225.
  qcoords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
  qweights=[qw5a,qw5b,qw5c,qw5b,qw5a]
 case _ :
  exit('unknown nqperdim')

nqel=nqperdim**ndim
nq=nqel*nel

###############################################################################
#@@ open output files & write headers
###############################################################################

vrms_file=open('OUTPUT/vrms.ascii',"w") ; vrms_file.write("#time,vrms\n")
pstats_file=open('OUTPUT/stats_pressure.ascii',"w") 
pstats_file.write("#istep,min p, max p\n")
vstats_file=open('OUTPUT/stats_velocity.ascii',"w") 
vstats_file.write("#istep,min(u),max(u),min(v),max(v)\n")
vstats_file.write("# "+vel_unit+"\n")
dt_file=open('OUTPUT/dt.ascii',"w") ; dt_file.write("#time dt1 dt2 dt\n") ; dt_file.write('#'+time_unit+'\n')
ptcl_stats_file=open('OUTPUT/stats_particle.ascii',"w")
timings_file=open('timings.ascii',"w")
TM_file=open('OUTPUT/total_mass.ascii',"w") 
EK_file=open('OUTPUT/kinetic_energy.ascii',"w") 
TVD_file=open('OUTPUT/viscous_dissipation.ascii',"w") 
pvd_solution_file=open('OUTPUT/solution.pvd',"w")
pvd_swarm_file=open('OUTPUT/swarm.pvd',"w")
mats_file=open('OUTPUT/stats_mats.ascii','w')
if solve_T:
   corner_q_file=open('OUTPUT/corner_heat_flux.ascii','w')
   Tstats_file=open('OUTPUT/stats_temperature.ascii',"w") 
   Nu_file=open('OUTPUT/Nu.ascii',"w") ; Nu_file.write("#time Nu\n")
   avrg_T_bot_file=open('OUTPUT/avrg_T_bot.ascii',"w") 
   avrg_T_top_file=open('OUTPUT/avrg_T_top.ascii',"w") 
   avrg_dTdz_bot_file=open('OUTPUT/avrg_dTdz_bot.ascii',"w") 
   avrg_dTdz_top_file=open('OUTPUT/avrg_dTdz_top.ascii',"w") 

###############################################################################

if nstep==1: CFLnb=0

if geometry=='box': volume=Lx*Lz
if geometry=='quarter': volume=np.pi*(Router**2-Rinner**2)/4
if geometry=='half': volume=np.pi*(Router**2-Rinner**2)/2
if geometry=='eighth': volume=np.pi*(Router**2-Rinner**2)/8

print('experiment=',experiment)
print('axisymmetric=',axisymmetric)
print('geometry=',geometry)
print('straighten_edges=',straighten_edges)
print('remove_rho_profile=',remove_rho_profile)
print('nelx,nelz=',nelx,nelz)
print('Lx,Lz=',Lx,Lz)
print('hx,hz=',hx,hz)
print('nn_V=',nn_V,'| nn_P=',nn_P,'| nel=',nel)
print('Nfem_V=',Nfem_V,'| Nfem_P=',Nfem_P,'| Nfem=',Nfem)
print('nqperdim=',nqperdim)
print('CFLnb=',CFLnb)
print('debug_ascii:',debug_ascii,'| debug_nan:',debug_nan)
print('solve_T:',solve_T)
print('tol_ss=',tol_ss)
print('end_time=',end_time/time_scale,time_unit)
print('averaging:',averaging)
print('formulation:',formulation)
print('particle_distribution=',particle_distribution)
print('RKorder=',RKorder)
print('nparticle_per_dim=',nparticle_per_dim)
print('nparticle=',nparticle)
print('every_solution_vtu',every_solution_vtu)
print('every_swarm_vtu',every_swarm_vtu)
print('every_quadpoints_vtu',every_quadpoints_vtu)
print('rho_DT_top',rho_DT_top)
print('rho_DT_bot',rho_DT_bot)
print('gravity_npts=',gravity_npts)  
print('top_free_slip=',top_free_slip,'| bot_free_slip=',bot_free_slip)
if geometry=='quarter' or geometry=='half' or geometry=='eighth':
   print('Rinner,Router=',Rinner,Router)
   print('hrad=',hrad)
print('-----------------------------')

###############################################################################
#@@ build velocity nodes coordinates 
# BL: bottom left, BR: bottom right, TL: top left, TR: top right
# if geometry is 'eighth', 'quarter' or 'half' we still need to set Lx=Lz=1
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64) 
z_V=np.zeros(nn_V,dtype=np.float64)
rad_V=np.zeros(nn_V,dtype=np.float64) 
theta_V=np.zeros(nn_V,dtype=np.float64)
top_Vnodes=np.zeros(nn_V,dtype=bool)
bot_Vnodes=np.zeros(nn_V,dtype=bool)
left_Vnodes=np.zeros(nn_V,dtype=bool)
right_Vnodes=np.zeros(nn_V,dtype=bool)
middleH_nodes=np.zeros(nn_V,dtype=bool)
middleV_nodes=np.zeros(nn_V,dtype=bool)
hull_nodes=np.zeros(nn_V,dtype=bool)

nnx=2*nelx+1 
nnz=2*nelz+1 

counter=0    
for j in range(0,2*nelz+1):
    for i in range(0,2*nelx+1):
        x_V[counter]=i*hx/2
        z_V[counter]=j*hz/2
        if (i==0): left_Vnodes[counter]=True
        if (i==2*nelx): right_Vnodes[counter]=True
        if (j==0): bot_Vnodes[counter]=True
        if (j==2*nelz): top_Vnodes[counter]=True
        if top_Vnodes[counter] or bot_Vnodes[counter] or\
           right_Vnodes[counter] or left_Vnodes[counter]: hull_nodes[counter]=True
        if abs(x_V[counter]/Lx-0.5)<eps: middleV_nodes[counter]=True
        if abs(z_V[counter]/Lz-0.5)<eps: middleH_nodes[counter]=True
        if i==0 and j==0: cornerBL=counter
        if i==nnx-1 and j==0: cornerBR=counter
        if i==0 and j==nnz-1: cornerTL=counter
        if i==nnx-1 and j==nnz-1: cornerTR=counter
        counter+=1
    #end for
#end for

if geometry=='quarter' or geometry=='half' or geometry=='eighth':
   for i in range(0,nn_V):
       rad_V[i]=Rinner+z_V[i]*(Router-Rinner)
       theta_V[i]=np.pi/2-x_V[i]*opening_angle
       x_V[i]=rad_V[i]*np.cos(theta_V[i])
       z_V[i]=rad_V[i]*np.sin(theta_V[i])

if debug_ascii: np.savetxt('DEBUG/mesh_V.ascii',np.array([x_V,z_V]).T,header='# x,z')

print("build V grid: %.3f s" % (clock.time() - start))

###############################################################################
#@@ connectivity for velocity nodes
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)
top_element=np.zeros(nel,dtype=bool)
bot_element=np.zeros(nel,dtype=bool)
left_element=np.zeros(nel,dtype=bool)
right_element=np.zeros(nel,dtype=bool)

counter=0
for j in range(0,nelz):
    for i in range(0,nelx):
        icon_V[0,counter]=i*2+1+j*2*nnx -1
        icon_V[1,counter]=i*2+3+j*2*nnx -1
        icon_V[2,counter]=i*2+3+j*2*nnx+nnx*2 -1
        icon_V[3,counter]=i*2+1+j*2*nnx+nnx*2 -1
        icon_V[4,counter]=i*2+2+j*2*nnx -1
        icon_V[5,counter]=i*2+3+j*2*nnx+nnx -1
        icon_V[6,counter]=i*2+2+j*2*nnx+nnx*2 -1
        icon_V[7,counter]=i*2+1+j*2*nnx+nnx -1
        icon_V[8,counter]=i*2+2+j*2*nnx+nnx -1
        if (i==0): left_element[counter]=True
        if (i==nelx-1): right_element[counter]=True
        if (j==0): bot_element[counter]=True
        if (j==nelz-1): top_element[counter]=True
        counter+=1
    #end for
#end for

print("build icon_V: %.3f s" % (clock.time()-start))

###############################################################################
# FIX MESH 
# 3 6 2
# 7 8 5
# 0-4-1
###############################################################################
# in the case of a curved axisymmetric domain, it could be beneficial to 
# straighten the element sides
###############################################################################

if axisymmetric and straighten_edges and \
   (geometry=='quarter' or geometry=='half' or geometry=='eighth'):
   for iel in range(0,nel):
       x_V[icon_V[4,iel]]=0.5*(x_V[icon_V[0,iel]]+x_V[icon_V[1,iel]])
       z_V[icon_V[4,iel]]=0.5*(z_V[icon_V[0,iel]]+z_V[icon_V[1,iel]])
       x_V[icon_V[5,iel]]=0.5*(x_V[icon_V[1,iel]]+x_V[icon_V[2,iel]])
       z_V[icon_V[5,iel]]=0.5*(z_V[icon_V[1,iel]]+z_V[icon_V[2,iel]])
       x_V[icon_V[6,iel]]=0.5*(x_V[icon_V[2,iel]]+x_V[icon_V[3,iel]])
       z_V[icon_V[6,iel]]=0.5*(z_V[icon_V[2,iel]]+z_V[icon_V[3,iel]])
       x_V[icon_V[7,iel]]=0.5*(x_V[icon_V[3,iel]]+x_V[icon_V[0,iel]])
       z_V[icon_V[7,iel]]=0.5*(z_V[icon_V[3,iel]]+z_V[icon_V[0,iel]])
       x_V[icon_V[8,iel]]=0.25*(x_V[icon_V[0,iel]]+x_V[icon_V[1,iel]]+x_V[icon_V[2,iel]]+x_V[icon_V[3,iel]])
       z_V[icon_V[8,iel]]=0.25*(z_V[icon_V[0,iel]]+z_V[icon_V[1,iel]]+z_V[icon_V[2,iel]]+z_V[icon_V[3,iel]])

###############################################################################
#@@ build pressure grid 
###############################################################################
start=clock.time()

x_P=np.zeros(nn_P,dtype=np.float64)
z_P=np.zeros(nn_P,dtype=np.float64)
rad_P=np.zeros(nn_P,dtype=np.float64) 
theta_P=np.zeros(nn_P,dtype=np.float64)
top_Pnodes=np.zeros(nn_P,dtype=bool)
bot_Pnodes=np.zeros(nn_P,dtype=bool)

counter=0    
for j in range(0,nelz+1):
    for i in range(0,nelx+1):
        x_P[counter]=i*hx
        z_P[counter]=j*hz
        if (j==0): bot_Pnodes[counter]=True
        if (j==nelz): top_Pnodes[counter]=True
        counter+=1
    #end for
 #end for

if geometry=='quarter' or geometry=='half' or geometry=='eighth':
   for i in range(0,nn_P):
       rad_P[i]=Rinner+z_P[i]*(Router-Rinner)
       theta_P[i]=np.pi/2-x_P[i]*opening_angle
       x_P[i]=rad_P[i]*np.cos(theta_P[i])
       z_P[i]=rad_P[i]*np.sin(theta_P[i])

if debug_ascii: np.savetxt('DEBUG/mesh_P.ascii',np.array([x_P,z_P]).T,header='# x,z')

print("build P grid: %.3f s" % (clock.time() - start))

###############################################################################
#@@ build pressure connectivity array 
###############################################################################
start=clock.time()

icon_P=np.zeros((m_P,nel),dtype=np.int32)

counter=0
for j in range(0,nelz):
    for i in range(0,nelx):
        icon_P[0,counter]=i+j*(nelx+1)
        icon_P[1,counter]=i+1+j*(nelx+1)
        icon_P[2,counter]=i+1+(j+1)*(nelx+1)
        icon_P[3,counter]=i+(j+1)*(nelx+1)
        counter+=1
    #end for
#end for

print("build icon_P: %.3f s" % (clock.time()-start))

###############################################################################
#@@ define velocity boundary conditions
# bc_fix_V is a vector of booleans of size Nfem_V
# bc_val_V is a vector of float64 of size Nfem_V
###############################################################################
start=clock.time()

bc_fix_V,bc_val_V=\
assign_boundary_conditions_V(x_V,z_V,rad_V,theta_V,ndof_V,Nfem_V,nn_V,\
                             hull_nodes,top_Vnodes,bot_Vnodes,left_Vnodes,right_Vnodes)

print("velocity b.c.: %.3f s" % (clock.time()-start))

###############################################################################
#@@ define temperature boundary conditions
# bc_fix_T is a vector of booleans of size Nfem_T
# bc_val_T is a vector of float64 of size Nfem_T
###############################################################################
start=clock.time()

if solve_T:
   bc_fix_T,bc_val_T=assign_boundary_conditions_T(x_V,z_V,rad_V,theta_V,Nfem_T,nn_V)
else:
   bc_fix_T=False
   
print("temperature b.c.: %.3f s" % (clock.time()-start))

###############################################################################
#@@ initial temperature. T is a vector of float64 of size nn_V
# Even if solve_T=False it needs to be allocated
###############################################################################
start=clock.time()

if solve_T: 
   T=initial_temperature(x_V,z_V,rad_V,theta_V,nn_V)

   for i in range(nn_V):
       if bc_fix_T[i]:
          T[i]=bc_val_T[i]

   T_mem=T.copy()

   if debug_ascii: np.savetxt('DEBUG/T_init.ascii',np.array([x_V,z_V,T]).T,header='# x,z,T')

   print("     -> T init (m,M) %.3e %.3e " %(np.min(T)-TKelvin,np.max(T)-TKelvin))

   print("initial temperature: %.3f s" % (clock.time()-start))

else:
   T=np.zeros(nn_V,dtype=np.float64) 

###############################################################################
#@@ define_mapping 
###############################################################################

#x_M,z_M,m_M=define_mapping(geometry,mapping,nelx,nel,x_V,z_V,icon_V,rad_V,theta_V)
#if debug_ascii: i
#np.savetxt('DEBUG/mesh_M.ascii',np.array([x_M.flatten(),z_M.flatten()]).T,header='# x,z')
#exit()

###############################################################################
#@@ compute area of elements / sanity check
#@@ precompute basis functions values at quadrature points
# JxWq is of size (nel,nqel)
###############################################################################
start=clock.time()

jcb=np.zeros((ndim,ndim),dtype=np.float64)
rq=np.zeros(nqel,dtype=np.float64) 
tq=np.zeros(nqel,dtype=np.float64) 
weightq=np.zeros(nqel,dtype=np.float64) 
N_V=np.zeros((nqel,m_V),dtype=np.float64) 
N_P=np.zeros((nqel,m_P),dtype=np.float64) 
dNdr_V=np.zeros((nqel,m_V),dtype=np.float64) 
dNdt_V=np.zeros((nqel,m_V),dtype=np.float64) 

area=np.zeros(nel,dtype=np.float64) 
x_e=np.zeros(nel,dtype=np.float64) 
z_e=np.zeros(nel,dtype=np.float64) 
rad_e=np.zeros(nel,dtype=np.float64) 
theta_e=np.zeros(nel,dtype=np.float64) 

JxWq=np.zeros((nel,nqel),dtype=np.float64) 
jcbi00q=np.zeros((nel,nqel),dtype=np.float64) 
jcbi01q=np.zeros((nel,nqel),dtype=np.float64) 
jcbi10q=np.zeros((nel,nqel),dtype=np.float64) 
jcbi11q=np.zeros((nel,nqel),dtype=np.float64) 

for iel in range(0,nel):
    counterq=0
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            if iel==0:
               rq[counterq]=qcoords[iq]
               tq[counterq]=qcoords[jq]
               weightq[counterq]=qweights[iq]*qweights[jq]
               N_V[counterq,0:m_V]=basis_functions_V(rq[counterq],tq[counterq])
               N_P[counterq,0:m_P]=basis_functions_P(rq[counterq],tq[counterq])
               dNdr_V[counterq,0:m_V]=basis_functions_V_dr(rq[counterq],tq[counterq])
               dNdt_V[counterq,0:m_V]=basis_functions_V_dt(rq[counterq],tq[counterq])
            #end if
            jcb[0,0]=np.dot(dNdr_V[counterq,:],x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V[counterq,:],z_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNdt_V[counterq,:],x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNdt_V[counterq,:],z_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq[iel,counterq]=np.linalg.det(jcb)*weightq[counterq]
            jcbi00q[iel,counterq]=jcbi[0,0]
            jcbi01q[iel,counterq]=jcbi[0,1]
            jcbi10q[iel,counterq]=jcbi[1,0]
            jcbi11q[iel,counterq]=jcbi[1,1]
            area[iel]+=JxWq[iel,counterq]
            counterq+=1
        #end for
    #end for
    x_e[iel]=x_V[icon_V[8,iel]]
    z_e[iel]=z_V[icon_V[8,iel]]
    if geometry=='quarter' or geometry=='half' or geometry=='eighth':
       rad_e[iel]=np.sqrt(x_e[iel]**2+z_e[iel]**2)
       theta_e[iel]=np.pi/2-np.arctan2(x_e[iel],z_e[iel])
#end for

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %e %e " %(area.sum(),volume))

print("comp elts areas, N, grad(N) at q pts: %.3f s" % (clock.time()-start))

###############################################################################
#@@ precompute basis functions and jacobian values at V nodes
###############################################################################
start=clock.time()

N_P_n=np.zeros((m_V,m_P),dtype=np.float64) 
dNdr_V_n=np.zeros((m_V,m_V),dtype=np.float64) 
dNdt_V_n=np.zeros((m_V,m_V),dtype=np.float64) 
jcbi00n=np.zeros((nel,m_V),dtype=np.float64) 
jcbi01n=np.zeros((nel,m_V),dtype=np.float64) 
jcbi10n=np.zeros((nel,m_V),dtype=np.float64) 
jcbi11n=np.zeros((nel,m_V),dtype=np.float64) 

for iel in range(0,nel):
    for i in range(0,m_V):
        if iel==0:
           N_P_n[i,0:m_P]=basis_functions_P(r_V[i],t_V[i])
           dNdr_V_n[i,0:m_V]=basis_functions_V_dr(r_V[i],t_V[i])
           dNdt_V_n[i,0:m_V]=basis_functions_V_dt(r_V[i],t_V[i])
        jcb[0,0]=np.dot(dNdr_V_n[i,:],x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr_V_n[i,:],z_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNdt_V_n[i,:],x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNdt_V_n[i,:],z_V[icon_V[:,iel]])
        jcbi=np.linalg.inv(jcb)
        jcbi00n[iel,i]=jcbi[0,0]
        jcbi01n[iel,i]=jcbi[0,1]
        jcbi10n[iel,i]=jcbi[1,0]
        jcbi11n[iel,i]=jcbi[1,1]

print("compute N & grad(N) at V nodes: %.3f s" % (clock.time()-start))

###############################################################################
#@@ compute coordinates of quadrature points - xq,zq are size (nel,nqel)
###############################################################################
start=clock.time()

xq=Q2_project_nodal_field_onto_qpoints(x_V,nqel,nel,N_V,icon_V)
zq=Q2_project_nodal_field_onto_qpoints(z_V,nqel,nel,N_V,icon_V)

print("     -> xq (m,M) %.3e %.3e " %(np.min(xq),np.max(xq)))
print("     -> zq (m,M) %.3e %.3e " %(np.min(zq),np.max(zq)))

if debug_ascii: np.savetxt('DEBUG/qpoints.ascii',np.array([xq.flatten(),zq.flatten()]).T,header='# x,z')

print("compute coords quad pts: %.3f s" % (clock.time()-start))

###############################################################################
#@@ compute gravity vector at quadrature points -  gxq,gzq are size (nel,nqel)
###############################################################################
start=clock.time()

gxq=np.zeros((nel,nqel),dtype=np.float64) 
gzq=np.zeros((nel,nqel),dtype=np.float64) 

for iel in range(0,nel):
    for iq in range(0,nqel):
        gxq[iel,iq],gzq[iel,iq]=gravity_model(xq[iel,iq],zq[iel,iq])

print("     -> gxq (m,M) %.3e %.3e " %(np.min(gxq),np.max(gxq)))
print("     -> gzq (m,M) %.3e %.3e " %(np.min(gzq),np.max(gzq)))

if debug_ascii: np.savetxt('DEBUG/qgravity.ascii',np.array([xq.flatten(),zq.flatten(),gxq.flatten(),gzq.flatten()]).T,header='#x,z,gx,gz')

print("compute grav at qpts: %.3f s" % (clock.time()-start))

###############################################################################
#@@ compute gravity on mesh points
###############################################################################
start=clock.time()

gx_n=np.zeros(nn_V,dtype=np.float64) 
gz_n=np.zeros(nn_V,dtype=np.float64) 
gx_e=np.zeros(nel,dtype=np.float64) 
gz_e=np.zeros(nel,dtype=np.float64) 

for i in range(0,nn_V):
    gx_n[i],gz_n[i]=gravity_model(x_V[i],z_V[i])

gr_n=gx_n*np.cos(theta_V)+gz_n*np.sin(theta_V)

for iel in range(0,nel):
    gx_e[iel],gz_e[iel]=gravity_model(x_e[iel],z_e[iel])

gr_e=gx_e*np.cos(theta_e)+gz_e*np.sin(theta_e)

if debug_ascii:
   np.savetxt('DEBUG/gr_n.ascii',np.array([x_V,z_V,gr_n]).T,header='#x,z,gr')
   np.savetxt('DEBUG/gr_e.ascii',np.array([x_e,z_e,gr_e]).T,header='#x,z,gr')

print("compute grav on nodes: %.3f s" % (clock.time()-start))

###############################################################################
#@@ compute normal vector of domain - NOT needed anymore
###############################################################################
#start=clock.time()
#nx,nz=compute_normals(geometry,nel,nn_V,nqel,m_V,icon_V,dNdr_V,dNdt_V,\
#                      JxWq,hull_nodes,jcbi00q,jcbi01q,jcbi10q,jcbi11q)
#if debug_ascii: np.savetxt('DEBUG/normal_vector.ascii',np.array([x_V[hull_nodes],\
#                           z_V[hull_nodes],nx[hull_nodes],nz[hull_nodes]]).T,header='#x,z,nx,nz')
#
#print("compute normal vector: %.3f s" % (clock.time()-start))

###############################################################################
#@@ compute array for assembly
###############################################################################
start=clock.time()

local_to_globalV=np.zeros((ndof_V_el,nel),dtype=np.int32)

for iel in range(0,nel):
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1+i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            local_to_globalV[ikk,iel]=m1
                 
print("compute local_to_globalV: %.3f s" % (clock.time()-start))

###############################################################################
#@@ fill II_V,JJ_V arrays for Stokes matrix
###############################################################################
start=clock.time()

bignb_V=nel*( (m_V*ndof_V)**2 + 2*(m_V*ndof_V*m_P) )

II_V=np.zeros(bignb_V,dtype=np.int32)    
JJ_V=np.zeros(bignb_V,dtype=np.int32)    

counter=0
for iel in range(0,nel):
    for ikk in range(ndof_V_el):
        m1=local_to_globalV[ikk,iel]
        for jkk in range(ndof_V_el):
            m2=local_to_globalV[jkk,iel]
            II_V[counter]=m1
            JJ_V[counter]=m2
            counter+=1
        for jkk in range(0,m_P):
            m2 =icon_P[jkk,iel]+Nfem_V
            II_V[counter]=m1
            JJ_V[counter]=m2
            counter+=1
            II_V[counter]=m2
            JJ_V[counter]=m1
            counter+=1

print("fill II_V,JJ_V arrays: %.3f s" % (clock.time()-start))

###############################################################################
#@@ fill II_T,JJ_T arrays for temperature matrix & plith matrix
###############################################################################
start=clock.time()

bignb_T=nel*m_T**2 

II_T=np.zeros(bignb_T,dtype=np.int32)    
JJ_T=np.zeros(bignb_T,dtype=np.int32)    

counter=0
for iel in range(0,nel):
    for ikk in range(m_T):
        m1=icon_V[ikk,iel]
        for jkk in range(m_T):
            m2=icon_V[jkk,iel]
            II_T[counter]=m1
            JJ_T[counter]=m2
            counter+=1

print("fill II_T,JJ_T arrays: %.3f s" % (clock.time()-start))

###############################################################################
#@@ particle coordinates setup
###############################################################################
start=clock.time()

swarm_x=np.zeros(nparticle,dtype=np.float64)
swarm_z=np.zeros(nparticle,dtype=np.float64)
swarm_r=np.zeros(nparticle,dtype=np.float64)
swarm_t=np.zeros(nparticle,dtype=np.float64)
swarm_iel=np.zeros(nparticle,dtype=np.int32)

match(particle_distribution):

     case(0): # random
         counter=0
         for iel in range(0,nel):
             for im in range(0,nparticle_per_element):
                 r=random.uniform(-1.,+1)
                 t=random.uniform(-1.,+1)
                 N=basis_functions_V(r,t)
                 swarm_x[counter]=np.dot(N[:],x_V[icon_V[:,iel]])
                 swarm_z[counter]=np.dot(N[:],z_V[icon_V[:,iel]])
                 swarm_r[counter]=r
                 swarm_t[counter]=t
                 swarm_iel[counter]=iel
                 counter+=1
             #end for
         #end for

     case(1): # regular

         counter=0
         for iel in range(0,nel):
             for j in range(0,nparticle_per_dim):
                 for i in range(0,nparticle_per_dim):
                     r=-1.+i*2./nparticle_per_dim + 1./nparticle_per_dim
                     t=-1.+j*2./nparticle_per_dim + 1./nparticle_per_dim
                     N=basis_functions_V(r,t)
                     swarm_x[counter]=np.dot(N[:],x_V[icon_V[:,iel]])
                     swarm_z[counter]=np.dot(N[:],z_V[icon_V[:,iel]])
                     swarm_r[counter]=r
                     swarm_t[counter]=t
                     swarm_iel[counter]=iel
                     counter+=1
                 #end for
             #end for
         #end for

     case(2): # Poisson Disc

         if geometry!='box': exit('Poisson disc not available with this geometry')

         kpoisson=30
         nparticle_wish=nel*nparticle_per_element # target
         print ('     -> nparticle_wish: %d ' % (nparticle_wish) )
         avrgdist=np.sqrt(Lx*Lz/nparticle_wish)/1.25
         nparticle,swarm_x,swarm_z=PoissonDisc(kpoisson,avrgdist,Lx,Lz)
         print ('     -> nparticle: %d ' % (nparticle) )

         swarm_r,swarm_t,swarm_iel=\
         locate_particles(nparticle,swarm_x,swarm_z,hx,hz,x_V,z_V,icon_V,nelx)

     case 3 : # pseudo-random

         counter=0
         for iel in range(0,nel):
             for j in range(0,nparticle_per_dim):
                 for i in range(0,nparticle_per_dim):
                     r=-1.+i*2./nparticle_per_dim + 1./nparticle_per_dim
                     t=-1.+j*2./nparticle_per_dim + 1./nparticle_per_dim
                     r+=random.uniform(-0.2,+0.2)*(2/nparticle_per_dim)
                     t+=random.uniform(-0.2,+0.2)*(2/nparticle_per_dim)
                     N=basis_functions_V(r,t)
                     swarm_x[counter]=np.dot(N[:],x_V[icon_V[:,iel]])
                     swarm_z[counter]=np.dot(N[:],z_V[icon_V[:,iel]])
                     swarm_r[counter]=r
                     swarm_t[counter]=t
                     swarm_iel[counter]=iel
                     counter+=1
                 #end for
             #end for
         #end for

     case _ :
         exit('unknown particle_distribution')

if debug_ascii: np.savetxt('DEBUG/swarm_distribution.ascii',np.array([swarm_x,swarm_z]).T,header='#x,z')

swarm_active=np.zeros(nparticle,dtype=bool) ; swarm_active[:]=True

print("     -> nparticle %d " % nparticle)
print("     -> swarm_x (m,M) %.3e %.3e " %(np.min(swarm_x),np.max(swarm_x)))
print("     -> swarm_z (m,M) %.3e %.3e " %(np.min(swarm_z),np.max(swarm_z)))

if geometry=='quarter' or geometry=='half' or geometry=='eighth':
   swarm_rad=np.sqrt(swarm_x**2+swarm_z**2)
   swarm_theta=np.pi/2-np.arctan2(swarm_x,swarm_z)
   print("     -> swarm_rad (m,M) %.3e %.3e " %(np.min(swarm_rad),np.max(swarm_rad)))
   print("     -> swarm_theta (m,M) %.3e %.3e " %(np.min(swarm_theta),np.max(swarm_theta)))
else:
   swarm_rad=0
   swarm_theta=0

swarm_strain=np.zeros(nparticle,dtype=np.float64)

print("particles setup: %.3f s" % (clock.time()-start))

###############################################################################
#@@ particle paint
###############################################################################
start=clock.time()

swarm_paint=np.zeros(nparticle,dtype=np.int32)

match(geometry):
 case 'box' :
  for i in [0,2,4,6,8,10,12,14]:
      dx=Lx/16
      for ip in range (0,nparticle):
          if swarm_x[ip]>i*dx and swarm_x[ip]<(i+1)*dx:
             swarm_paint[ip]+=1
  for i in [0,2,4,6,8,10,12,14]:
      dz=Lz/16
      for ip in range (0,nparticle):
          if swarm_z[ip]>i*dz and swarm_z[ip]<(i+1)*dz:
             swarm_paint[ip]+=1

 case 'quarter' | 'half' | 'eighth':
  for i in [0,2,4,6,8,10,12,14]:
      drad=(Router-Rinner)/16
      for ip in range (0,nparticle):
          if swarm_rad[ip]>Rinner+i*drad and swarm_rad[ip]<Rinner+(i+1)*drad:
             swarm_paint[ip]+=1
  for i in [0,2,4,6,8,10,12,14]:
      dtheta=opening_angle/16
      for ip in range (0,nparticle):
          if swarm_theta[ip]>theta_min+i*dtheta and swarm_theta[ip]<theta_min+(i+1)*dtheta:
             swarm_paint[ip]+=1

print("particles paint: %.3f s" % (clock.time()-start))

###############################################################################
#@@ particle layout
###############################################################################
start=clock.time()

swarm_mat=particle_layout(nparticle,swarm_x,swarm_z,swarm_rad,swarm_theta,Lx,Lz)

print("     -> swarm_mat (m,M) %d %d " %(np.min(swarm_mat),np.max(swarm_mat)))
    
if debug_ascii: np.savetxt('DEBUG/swarm_mat.ascii',np.array([swarm_x,swarm_z,swarm_mat]).T,header='#x,z,mat')

if use_melting:
   swarm_F=np.zeros(nparticle,dtype=np.float32)
   swarm_sst=np.zeros(nparticle,dtype=np.float32)
else:
   swarm_F=0.
   swarm_sst=0.

print("particle layout: %.3f s" % (clock.time()-start))

###############################################################################
###############################################################################
###############################################################################
#@@ --------------------- time stepping loop ----------------------------------
###############################################################################
###############################################################################
###############################################################################

geological_time=0.
dt1_mem=1e50
dt2_mem=1e50
       
exx_n=np.zeros(nn_V,dtype=np.float64)  
ezz_n=np.zeros(nn_V,dtype=np.float64)  
exz_n=np.zeros(nn_V,dtype=np.float64)  
dpdx_n=np.zeros(nn_V,dtype=np.float64)  
dpdz_n=np.zeros(nn_V,dtype=np.float64)  
u_mem=np.zeros(nn_V,dtype=np.float64)  
w_mem=np.zeros(nn_V,dtype=np.float64)  
p_mem=np.zeros(nn_P,dtype=np.float64)  
q=np.zeros(nn_V,dtype=np.float64)  

topstart=clock.time()

for istep in range(0,nstep):
    print("-------------------------------------")
    print("istep= %d | time= %.4e " %(istep,geological_time/time_scale))
    print("-------------------------------------")

    ###############################################################################################
    #@@ interpolate strain rate, pressure and temperature on particles
    ###############################################################################################
    start=clock.time()

    swarm_exx=interpolate_field_on_particles(nparticle,swarm_r,swarm_t,swarm_iel,exx_n,icon_V)
    swarm_ezz=interpolate_field_on_particles(nparticle,swarm_r,swarm_t,swarm_iel,ezz_n,icon_V)
    swarm_exz=interpolate_field_on_particles(nparticle,swarm_r,swarm_t,swarm_iel,exz_n,icon_V)

    swarm_p=interpolate_field_on_particles(nparticle,swarm_r,swarm_t,swarm_iel,q,icon_V)

    if solve_T:
       swarm_T=interpolate_field_on_particles(nparticle,swarm_r,swarm_t,swarm_iel,T,icon_V)
       print("     -> swarm_T (m,M) %.3e %.3e " %(np.min(swarm_T)-TKelvin,np.max(swarm_T)-TKelvin))
    else:
       swarm_T=0

    print("interp sr, q, T on particles: %.3fs" % (clock.time()-start)) ; timings[24]+=clock.time()-start

    ###############################################################################################
    #@@ compute depletion and super solidus temperature
    ###############################################################################################
    start=clock.time()

    if istep>0 and use_melting and solve_T: 
       swarm_F,swarm_sst=update_F(nparticle,swarm_p,swarm_T,swarm_F)

       print('****************************************')

       print("     -> swarm_F (m,M) %.3e %.3e " %(np.min(swarm_F),np.max(swarm_F)))
       print("     -> swarm_sst (m,M) %.3e %.3e " %(np.min(swarm_sst),np.max(swarm_sst)))

    print("melting on particles: %.3fs" % (clock.time()-start)) 

    ###############################################################################################
    #@@ evaluate density and viscosity on particles (and hcond, hcapa, hprod)
    # if solve_T is false then swarm_{hcond,hcapa,hprod} are scalars equal to zero
    ###############################################################################################
    start=clock.time()

    swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod=\
    material_model(nparticle,swarm_mat,swarm_x,swarm_z,swarm_rad,swarm_theta,\
                   swarm_exx,swarm_ezz,swarm_exz,swarm_T,swarm_p) 

    print("     -> swarm_rho (m,M) %.5e %.5e " %(np.min(swarm_rho),np.max(swarm_rho)))
    print("     -> swarm_eta (m,M) %.5e %.5e " %(np.min(swarm_eta),np.max(swarm_eta)))

    if solve_T:
       print("     -> swarm_hcapa (m,M) %.5e %.5e " %(np.min(swarm_hcapa),np.max(swarm_hcapa)))
       print("     -> swarm_hcond (m,M) %.5e %.5e " %(np.min(swarm_hcond),np.max(swarm_hcond)))
       print("     -> swarm_hprod (m,M) %.5e %.5e " %(np.min(swarm_hprod),np.max(swarm_hprod)))

    if debug_ascii: np.savetxt('DEBUG/swarm_rho.ascii',np.array([swarm_x,swarm_z,swarm_rho]).T,header='# x,z,rho')
    if debug_ascii: np.savetxt('DEBUG/swarm_eta.ascii',np.array([swarm_x,swarm_z,swarm_eta]).T,header='# x,z,eta')

    print("call material model on particles: %.3fs" % (clock.time()-start)) ; timings[15]+=clock.time()-start

    ###############################################################################################
    #@@ project particle properties on elements 
    # this is also where the nparticle_e array is filled
    ###############################################################################################
    start=clock.time()

    rho_e,eta_e,nparticle_e=\
    project_particles_on_elements(nel,nparticle,swarm_rho,swarm_eta,swarm_iel,averaging)

    if np.min(nparticle_e)==0: 
       exit('ABORT: an element contains no particle!')

    ptcl_stats_file.write("%d %d %d\n" % (istep,np.min(nparticle_e),\
                                                np.max(nparticle_e)))
    ptcl_stats_file.flush()

    print("     -> rho_e (m,M) %.3e %.3e " %(np.min(rho_e),np.max(rho_e)))
    print("     -> eta_e (m,M) %.3e %.3e " %(np.min(eta_e),np.max(eta_e)))

    if debug_ascii: np.savetxt('DEBUG/rho_e.ascii',np.array([x_e,z_e,rho_e]).T,header='# x,z,rho')
    if debug_ascii: np.savetxt('DEBUG/eta_e.ascii',np.array([x_e,z_e,eta_e]).T,header='# x,z,eta')

    if debug_nan and np.isnan(np.sum(rho_e)): exit('nan found in rho_e')
    if debug_nan and np.isnan(np.sum(eta_e)): exit('nan found in eta_e')

    print("project particle fields on elements: %.3fs" % (clock.time()-start)) ; timings[17]+=clock.time()-start

    ###############################################################################################
    #@@ project particle properties on V nodes
    # nodal rho & eta are computed on nodes 0,1,2,3, while values on nodes
    # 4,5,6,7,8 are obtained by simple averages. In the end we obtaine Q1 fields 
    ###############################################################################################
    start=clock.time()

    rho_n=project_particle_field_on_nodes(nel,nn_V,nparticle,swarm_rho,icon_V,swarm_iel,swarm_r,swarm_t,'arithmetic')
    eta_n=project_particle_field_on_nodes(nel,nn_V,nparticle,swarm_eta,icon_V,swarm_iel,swarm_r,swarm_t,averaging)

    print("     -> rho_n (m,M) %.3e %.3e " %(np.min(rho_n),np.max(rho_n)))
    print("     -> eta_n (m,M) %.3e %.3e " %(np.min(eta_n),np.max(eta_n)))

    if debug_ascii: np.savetxt('DEBUG/rho_n.ascii',np.array([x_V,z_V,rho_n,rad_V,theta_V]).T,header='# x,z,rho,rad,theta')
    if debug_ascii: np.savetxt('DEBUG/eta_n.ascii',np.array([x_V,z_V,eta_n,rad_V,theta_V]).T,header='# x,z,eta,rad,theta')

    if solve_T:
       hcond_n=project_particle_field_on_nodes(nel,nn_V,nparticle,swarm_hcond,icon_V,swarm_iel,'arithmetic')
       hcapa_n=project_particle_field_on_nodes(nel,nn_V,nparticle,swarm_hcapa,icon_V,swarm_iel,'arithmetic')
       hprod_n=project_particle_field_on_nodes(nel,nn_V,nparticle,swarm_hprod,icon_V,swarm_iel,'arithmetic')

       print("     -> hcond_n (m,M) %.3e %.3e " %(np.min(hcond_n),np.max(hcond_n)))
       print("     -> hcapa_n (m,M) %.3e %.3e " %(np.min(hcapa_n),np.max(hcapa_n)))
       print("     -> hprod_n (m,M) %.3e %.3e " %(np.min(hprod_n),np.max(hprod_n)))

       if debug_ascii: np.savetxt('DEBUG/hcond_n.ascii',np.array([x_V,z_V,hcond_n]).T,header='# x,z,hcond')
       if debug_ascii: np.savetxt('DEBUG/hcapa_n.ascii',np.array([x_V,z_V,hcapa_n]).T,header='# x,z,hcapa')
       if debug_ascii: np.savetxt('DEBUG/hprod_n.ascii',np.array([x_V,z_V,hprod_n]).T,header='# x,z,hprod')

    print("project particle fields on nodes: %.3fs" % (clock.time()-start)) ; timings[18]+=clock.time()-start

    ###########################################################################
    # compute (nodal) rho profile
    ###########################################################################
    start=clock.time()

    rho_n_profile=np.zeros(nnz,dtype=np.float64)
    rho_e_profile=np.zeros(nelz,dtype=np.float64)

    counter=0    
    for j in range(0,nnz):
        for i in range(0,nnx):
            rho_n_profile[j]+=rho_n[counter]
            counter+=1
    rho_n_profile/=nnx

    counter=0
    for j in range(0,nelz):
        for i in range(0,nelx):
            rho_e_profile[j]+=rho_e[counter]
            counter+=1
    rho_e_profile/=nelx

    if geometry=='box':
       np.savetxt('OUTPUT/rho_n_profile.ascii',np.array([z_V[left_Vnodes],rho_n_profile]).T,header='# z,rho')
       np.savetxt('OUTPUT/rho_e_profile.ascii',np.array([z_e[left_element],rho_e_profile]).T,header='# z,rho')
    else:
       np.savetxt('OUTPUT/rho_n_profile.ascii',np.array([rad_V[left_Vnodes],rho_n_profile]).T,header='# rad,rho')
       np.savetxt('OUTPUT/rho_e_profile.ascii',np.array([rad_e[left_element],rho_e_profile]).T,header='# rad,rho')

    print("     -> rho_n_profile (m,M) %.3e %.3e " %(np.min(rho_n_profile),np.max(rho_n_profile)))
    print("     -> rho_e_profile (m,M) %.3e %.3e " %(np.min(rho_e_profile),np.max(rho_e_profile)))

    print("compute rho_profile: %.3fs" % (clock.time()-start)) 

    ###########################################################################
    #@@ remove nodal rho profile
    ###########################################################################
    start=clock.time()

    if remove_rho_profile:
       rho_DT_top-=rho_n_profile[nnz-1]
       rho_DT_bot-=rho_n_profile[0]

       counter=0    
       for j in range(0,nnz):
           for i in range(0,nnx):
               rho_n[counter]-=rho_n_profile[j]
               counter+=1

       counter=0    
       for j in range(0,nelz):
           for i in range(0,nelx):
               rho_e[counter]-=rho_e_profile[j]
               counter+=1

    print("remove rho_profile: %.3fs" % (clock.time()-start)) 

    ###########################################################################
    #@@ project nodal/elemental values onto quadrature points
    # rhoq, etaq, exxq, ezzq, exzq, hcondq, hcapaq, hprodq have size (nel,nqel)
    ###########################################################################
    start=clock.time()

    if use_elemental_rho:
       rhoq=np.zeros((nel,nqel),dtype=np.float64)
       for iel in range(0,nel):
           rhoq[iel,:]=rho_e[iel]
    else:
       rhoq=Q1_project_nodal_field_onto_qpoints(rho_n,nqel,nel,N_P,icon_V)

    if use_elemental_eta:
       etaq=np.zeros((nel,nqel),dtype=np.float64)
       for iel in range(0,nel):
           etaq[iel,:]=eta_e[iel]
    else:
       etaq=Q1_project_nodal_field_onto_qpoints(eta_n,nqel,nel,N_P,icon_V)

    exxq=Q2_project_nodal_field_onto_qpoints(exx_n,nqel,nel,N_V,icon_V)
    ezzq=Q2_project_nodal_field_onto_qpoints(ezz_n,nqel,nel,N_V,icon_V)
    exzq=Q2_project_nodal_field_onto_qpoints(exz_n,nqel,nel,N_V,icon_V)
    dpdxq=Q2_project_nodal_field_onto_qpoints(dpdx_n,nqel,nel,N_V,icon_V)
    dpdzq=Q2_project_nodal_field_onto_qpoints(dpdz_n,nqel,nel,N_V,icon_V)

    if solve_T:
       Tq=Q2_project_nodal_field_onto_qpoints(T,nqel,nel,N_V,icon_V)
       hcapaq=Q1_project_nodal_field_onto_qpoints(hcapa_n,nqel,nel,N_P,icon_V)
       hcondq=Q1_project_nodal_field_onto_qpoints(hcond_n,nqel,nel,N_P,icon_V)
       hprodq=Q1_project_nodal_field_onto_qpoints(hprod_n,nqel,nel,N_P,icon_V)
    else:
       Tq=np.zeros((nel,nqel),dtype=np.float64)
       hcapaq=np.zeros((nel,nqel),dtype=np.float64)
       hcondq=np.zeros((nel,nqel),dtype=np.float64)
       hprodq=np.zeros((nel,nqel),dtype=np.float64)

    print("     -> rhoq (m,M) %.5e %.5e " %(np.min(rhoq),np.max(rhoq)))
    print("     -> etaq (m,M) %.5e %.5e " %(np.min(etaq),np.max(etaq)))

    if solve_T:
       print("     -> Tq (m,M) %.5e %.5e " %(np.min(Tq),np.max(Tq)))
       print("     -> hcapaq (m,M) %.5e %.5e " %(np.min(hcapaq),np.max(hcapaq)))
       print("     -> hcondq (m,M) %.5e %.5e " %(np.min(hcondq),np.max(hcondq)))
       print("     -> hprodq (m,M) %.5e %.5e " %(np.min(hprodq),np.max(hprodq)))

    if debug_ascii: np.savetxt('DEBUG/rhoq.ascii',np.array([xq.flatten(),zq.flatten(),rhoq.flatten()]).T,header='#x,z,rho')
    if debug_ascii: np.savetxt('DEBUG/etaq.ascii',np.array([xq.flatten(),zq.flatten(),etaq.flatten()]).T,header='#x,z,eta')
    if debug_ascii and solve_T: np.savetxt('DEBUG/Tq.ascii',np.array([xq.flatten(),zq.flatten(),Tq.flatten()]).T,header='#x,z,T')
    if debug_ascii and solve_T: np.savetxt('DEBUG/hcapaq.ascii',np.array([xq.flatten(),zq.flatten(),hcapaq.flatten()]).T,header='#x,z,hcapa')
    if debug_ascii and solve_T: np.savetxt('DEBUG/hcondq.ascii',np.array([xq.flatten(),zq.flatten(),hcondq.flatten()]).T,header='#x,z,hcond')
    if debug_ascii and solve_T: np.savetxt('DEBUG/hprodq.ascii',np.array([xq.flatten(),zq.flatten(),hprodq.flatten()]).T,header='#x,z,hprod')

    print("project nodal fields onto qpts: %.3fs" % (clock.time()-start)) ; timings[21]+=clock.time()-start

    ###########################################################################
    #@@ compute lithostatic pressure a la Jourdon & May, Solid Earth, 2022
    ###########################################################################
    start=clock.time()

    VV_T,rhs=build_matrix_plith(bignb_T,nel,nqel,m_T,Nfem_T,icon_V,rhoq,gxq,gzq,\
                                JxWq,N_V,dNdr_V,dNdt_V,jcbi00q,jcbi01q,jcbi10q,jcbi11q,top_Vnodes)
    sparse_matrix=sparse.coo_matrix((VV_T,(II_T,JJ_T)),shape=(Nfem_T,Nfem_T)).tocsr()
    plith=sps.linalg.spsolve(sparse_matrix,rhs)

    print("     -> plith (m,M) %.3e %.3e " %(np.min(plith),np.max(plith)))

    print("compute lithostatic pressure: %.3fs" % (clock.time()-start)) ; timings[28]+=clock.time()-start

    ###########################################################################
    #@@ build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    ###########################################################################
    start=clock.time()

    if solve_Stokes:
       VV_V,rhs=build_matrix_stokes(bignb_V,nel,nqel,m_V,m_P,ndof_V,Nfem_V,Nfem,\
                                    ndof_V_el,icon_V,icon_P,rhoq,etaq,JxWq,\
                                    local_to_globalV,gxq,gzq,N_V,N_P,dNdr_V,dNdt_V,\
                                    jcbi00q,jcbi01q,jcbi10q,jcbi11q,\
                                    eta_ref,L_ref,bc_fix_V,bc_val_V,\
                                    bot_element,top_element,bot_free_slip,top_free_slip,\
                                    geometry,theta_V,axisymmetric,xq)

    if debug_nan and np.isnan(np.sum(VV_V)): exit('nan found in VV_V')

    print("build FE matrix stokes: %.3fs" % (clock.time()-start)) ; timings[1]+=clock.time()-start

    ###############################################################################################
    #@@ solve stokes system
    ###############################################################################################
    start=clock.time()

    if solve_Stokes:
       sparse_matrix=sparse.coo_matrix((VV_V,(II_V,JJ_V)),shape=(Nfem,Nfem)).tocsr()
       sol=sps.linalg.spsolve(sparse_matrix,rhs)
    else:
       sol=np.zeros(Nfem,dtype=np.float64)

    print("solve time: %.3f s" % (clock.time()-start)) ; timings[2]+=clock.time()-start

    ###############################################################################################
    #@@ split solution into separate u,v,p velocity arrays
    ###############################################################################################
    start=clock.time()

    u,w=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
    p=sol[Nfem_V:Nfem]*(eta_ref/L_ref)

    if debug_nan and np.isnan(np.sum(u)): exit('nan found in u')
    if debug_nan and np.isnan(np.sum(w)): exit('nan found in w')
    if debug_nan and np.isnan(np.sum(p)): exit('nan found in p')

    if (geometry=='quarter' or geometry=='half' or geometry=='eighth') and top_free_slip:
       for i in range(0,nn_V):
           if top_Vnodes[i] and (not bc_fix_V[2*i]) and (not bc_fix_V[2*i+1]):
              ui=np.cos(theta_V[i])*u[i]-np.sin(theta_V[i])*w[i]
              wi=np.sin(theta_V[i])*u[i]+np.cos(theta_V[i])*w[i] 
              u[i]=ui
              w[i]=wi
                 
    if (geometry=='quarter' or geometry=='half' or geometry=='eighth') and bot_free_slip:
       for i in range(0,nn_V):
           if bot_Vnodes[i] and (not bc_fix_V[2*i]) and (not bc_fix_V[2*i+1]):
              ui=np.cos(theta_V[i])*u[i]-np.sin(theta_V[i])*w[i]
              wi=np.sin(theta_V[i])*u[i]+np.cos(theta_V[i])*w[i] 
              u[i]=ui
              w[i]=wi

    vel=np.sqrt(u**2+w**2)

    if geometry=='box':
       np.savetxt('OUTPUT/top_vel_'+str(istep)+'.ascii',np.array([x_V[top_Vnodes],vel[top_Vnodes]]).T)
       np.savetxt('OUTPUT/bot_vel_'+str(istep)+'.ascii',np.array([x_V[bot_Vnodes],vel[bot_Vnodes]]).T)
    if geometry=='quarter' or geometry=='half' or geometry=='eighth':
       np.savetxt('OUTPUT/top_vel_'+str(istep)+'.ascii',np.array([theta_V[top_Vnodes],vel[top_Vnodes]]).T)
       np.savetxt('OUTPUT/bot_vel_'+str(istep)+'.ascii',np.array([theta_V[bot_Vnodes],vel[bot_Vnodes]]).T)

    print("     -> u (m,M) %.3e %.3e %s" %(np.min(u)/vel_scale,np.max(u)/vel_scale,vel_unit))
    print("     -> w (m,M) %.3e %.3e %s" %(np.min(w)/vel_scale,np.max(w)/vel_scale,vel_unit))
    print("     -> p (m,M) %.3e %.3e %s" %(np.min(p)/p_scale,np.max(p)/p_scale,p_unit))

    vstats_file.write("%.3e %.3e %.3e %.3e %.3e\n" % (istep,np.min(u)/vel_scale,np.max(u)/vel_scale,\
                                                            np.min(w)/vel_scale,np.max(w)/vel_scale))
    vstats_file.flush()

    if debug_ascii: np.savetxt('DEBUG/velocity.ascii',np.array([x_V,z_V,u,w,rad_V,theta_V]).T,header='# x,z,u,w,rad,theta')
    if debug_ascii: np.savetxt('DEBUG/pressure.ascii',np.array([x_P,z_P,p,rad_P,theta_P]).T,header='# x,z,p,rad,theta')

    print("split vel into u,v: %.3f s" % (clock.time()-start)) ; timings[14]+=clock.time()-start

    ###############################################################################################
    #@@ convert velocity to polar coordinates
    ###############################################################################################

    if geometry=='quarter' or geometry=='half' or geometry=='eighth':
       if axisymmetric:
          vr=u*np.cos(theta_V)+w*np.sin(theta_V)
          vt=u*np.sin(theta_V)-w*np.cos(theta_V)
       else:
          vr= u*np.cos(theta_V)+w*np.sin(theta_V)
          vt=-u*np.sin(theta_V)+w*np.cos(theta_V)
       if debug_ascii: 
          np.savetxt('DEBUG/velocity_polar.ascii',np.array([x_V,z_V,vr,vt,rad_V,theta_V]).T,header='#x,z,vr,vt,rad,theta')
          np.savetxt('DEBUG/top_vt.ascii',np.array([theta_V[top_Vnodes],vt[top_Vnodes]]).T,header='#theta,vt')
          np.savetxt('DEBUG/bot_vt.ascii',np.array([theta_V[bot_Vnodes],vt[bot_Vnodes]]).T,header='#theta,vt')
    else:
       vr=0 ; vt=0
    
    ###########################################################################
    #@@ compute timestep
    # note that the timestep is not allowed to increase by more than 25% in one go
    ###########################################################################
    start=clock.time()

    if solve_Stokes:
       if geometry=='box': dt1=CFLnb*min(hx,hz)/np.max(vel)
       if geometry=='quarter': dt1=CFLnb*hrad/np.max(vel)
       if geometry=='half': dt1=CFLnb*hrad/np.max(vel)
       if geometry=='eighth': dt1=CFLnb*hrad/np.max(vel)
    else:
       dt1=0.

    print('     -> dt1= %.3e %s' %(dt1/time_scale,time_unit))
    
    if solve_T:
       avrg_hcond=np.average(swarm_hcond)
       avrg_hcapa=np.average(swarm_hcapa)
       avrg_rho=np.average(swarm_rho)
       if geometry=='box': dt2=CFLnb*min(hx,hz)**2/(avrg_hcond/avrg_hcapa/avrg_rho)
       if geometry=='quarter': dt2=CFLnb*hrad**2/(avrg_hcond/avrg_hcapa/avrg_rho)
       if geometry=='half': dt2=CFLnb*hrad**2/(avrg_hcond/avrg_hcapa/avrg_rho)
       if geometry=='eighth': dt2=CFLnb*hrad**2/(avrg_hcond/avrg_hcapa/avrg_rho)
       print('     -> dt2= %.3e %s' %(dt2/time_scale,time_unit))
    else:
       dt2=1e50

    dt1=min(dt1,1.25*dt1_mem) # limiter
    dt2=min(dt2,1.25*dt2_mem) # limiter

    dt=np.min([dt1,dt2,dt_max])

    geological_time+=dt

    print('     -> dt = %.3e %s' %(dt/time_scale,time_unit))
    print('     -> geological time = %e %s' %(geological_time/time_scale,time_unit))

    dt_file.write("%e %e %e %e\n" % (geological_time/time_scale,dt1/time_scale,dt2/time_scale,dt/time_scale)) 
    dt_file.flush()

    dt1_mem=dt1
    dt2_mem=dt2

    print("compute time step: %.3f s" % (clock.time()-start)) ; timings[19]+=clock.time()-start

    ###############################################################################################
    #@@ normalise pressure: simple approach to have avrg p = 0 (volume or surface)
    # note that the surface normalisation is not super clean
    ###############################################################################################
    start=clock.time()

    pressure_average=compute_pressure_average(geometry,pressure_normalisation,axisymmetric,top_element,\
                                              nel,nelx,nqel,N_P,JxWq,p,icon_P,theta_P,xq,volume)

    print('     -> pressure_average=',pressure_average)

    p-=pressure_average

    print("     -> p (m,M) %.3e %.3e %s" %(np.min(p),np.max(p),p_unit))

    pstats_file.write("%d %.3e %.3e\n" % (istep,np.min(p),np.max(p)))
    pstats_file.flush()

    if geometry=='box':
       np.savetxt('OUTPUT/top_p_'+str(istep)+'.ascii',np.array([x_P[top_Pnodes],p[top_Pnodes]]).T)
       np.savetxt('OUTPUT/bot_p_'+str(istep)+'.ascii',np.array([x_P[bot_Pnodes],p[bot_Pnodes]]).T)
    if geometry=='quarter' or geometry=='half' or geometry=='eighth':
       np.savetxt('OUTPUT/top_p_'+str(istep)+'.ascii',np.array([theta_P[top_Pnodes],p[top_Pnodes]]).T)
       np.savetxt('OUTPUT/bot_p_'+str(istep)+'.ascii',np.array([theta_P[bot_Pnodes],p[bot_Pnodes]]).T)

    if debug_ascii: np.savetxt('DEBUG/pressure_normalised.ascii',np.array([x_P,z_P,p,rad_P,theta_P]).T,header='# x,z,p,rad,theta')

    print("normalise pressure: %.3f s" % (clock.time()-start)) ; timings[12]+=clock.time()-start

    ###############################################################################################
    #@@ compute elemental pressure
    ###############################################################################################
    start=clock.time()

    p_e=np.zeros(nel,dtype=np.float64)  

    for iel in range(0,nel):
        p_e[iel]=np.sum(p[icon_P[:,iel]])/m_P

    print("     -> p_e (m,M) %.3e %.3e %s" %(np.min(p_e)/p_scale,np.max(p_e)/p_scale,p_unit))

    if geometry=='box':
       np.savetxt('OUTPUT/top_p_e_'+str(istep)+'.ascii',np.array([x_e[top_element],p_e[top_element]]).T)
       np.savetxt('OUTPUT/bot_p_e_'+str(istep)+'.ascii',np.array([x_e[bot_element],p_e[bot_element]]).T)
    if geometry=='quarter' or geometry=='half' or geometry=='eighth':
       np.savetxt('OUTPUT/top_p_e_'+str(istep)+'.ascii',np.array([theta_e[top_element],p_e[top_element]]).T)
       np.savetxt('OUTPUT/bot_p_e_'+str(istep)+'.ascii',np.array([theta_e[bot_element],p_e[bot_element]]).T)

    if debug_ascii: np.savetxt('DEBUG/pressure_e.ascii',np.array([x_e,z_e,p_e]).T,header='# x,z,p')

    print("compute elemental pressure: %.3f s" % (clock.time()-start)) #; timings[14]+=clock.time()-start

    ###############################################################################################
    #@@ project Q1 pressure onto Q2 (vel,T) mesh
    ###############################################################################################
    start=clock.time()

    q=compute_nodal_pressure(m_V,nn_V,icon_V,icon_P,p,N_P_n)
    
    print("     -> q (m,M) %.3e %.3e %s" %(np.min(q),np.max(q),p_unit))

    if debug_ascii: np.savetxt('DEBUG/q.ascii',np.array([x_V,z_V,q]).T,header='# x,z,q')

    if geometry=='box':
       np.savetxt('OUTPUT/top_q_'+str(istep)+'.ascii',np.array([x_V[top_Vnodes],q[top_Vnodes]]).T)
       np.savetxt('OUTPUT/bot_q_'+str(istep)+'.ascii',np.array([x_V[bot_Vnodes],q[bot_Vnodes]]).T)
    if geometry=='quarter' or geometry=='half' or geometry=='eighth':
       np.savetxt('OUTPUT/top_q_'+str(istep)+'.ascii',np.array([theta_V[top_Vnodes],q[top_Vnodes]]).T)
       np.savetxt('OUTPUT/bot_q_'+str(istep)+'.ascii',np.array([theta_V[bot_Vnodes],q[bot_Vnodes]]).T)

    print("compute nodal press: %.3f s" % (clock.time()-start)) ; timings[3]+=clock.time()-start

    ###############################################################################################
    #@@ project velocity on quadrature points
    ###############################################################################################
    start=clock.time()

    uq=Q2_project_nodal_field_onto_qpoints(u,nqel,nel,N_V,icon_V)
    wq=Q2_project_nodal_field_onto_qpoints(w,nqel,nel,N_V,icon_V)

    print("project vel on quad points: %.3f s" % (clock.time()-start)) ; timings[21]+=clock.time()-start

    ###############################################################################################
    #@@ build temperature matrix
    ###############################################################################################
    start=clock.time()

    if solve_T: 
       VV_T,rhs=build_matrix_energy(bignb_T,nel,nqel,m_T,Nfem_T,T,icon_V,rhoq,etaq,Tq,uq,wq,\
                                    hcondq,hcapaq,exxq,ezzq,exzq,dpdxq,dpdzq,JxWq,N_V,dNdr_V,dNdt_V,\
                                    jcbi00q,jcbi01q,jcbi10q,jcbi11q,\
                                    bc_fix_T,bc_val_T,dt,formulation,rho0)

       print("build FE matrix : %.3f s" % (clock.time()-start)) ; timings[4]+=clock.time()-start

    ###############################################################################################
    #@@ solve temperature system
    ###############################################################################################
    start=clock.time()

    if solve_T: 
       sparse_matrix=sparse.coo_matrix((VV_T,(II_T,JJ_T)),shape=(Nfem_T,Nfem_T)).tocsr()

       T=sps.linalg.spsolve(sparse_matrix,rhs)

       if debug_nan and np.isnan(np.sum(T)): exit('nan found in T')

       print("     -> T (m,M) %.3e %.3e " %(np.min(T),np.max(T)))

       if debug_ascii: np.savetxt('DEBUG/T.ascii',np.array([x_V,z_V,T]).T,header='# x,z,T')

       Tstats_file.write("%.3e %.3e %.3e\n" %(istep,np.min(T)-TKelvin,np.max(T)-TKelvin)) 
       Tstats_file.flush()

       print("solve T time: %.3f s" % (clock.time()-start)) ; timings[5]+=clock.time()-start

    #end if solve_T

    ###############################################################################################
    #@@ compute vrms 
    ###############################################################################################
    start=clock.time()

    vrms,EK,WAG,TVD,GPE,ITE,TM=\
    global_quantities(nel,nqel,xq,zq,uq,wq,Tq,rhoq,hcapaq,etaq,exxq,ezzq,exzq,volume,JxWq,gxq,gzq)

    vrms_file.write("%e %e \n" % (geological_time/time_scale,vrms/vel_scale)) ; vrms_file.flush()
    TM_file.write("%e %e \n" % (geological_time/time_scale,TM)) ; TM_file.flush()
    EK_file.write("%e %e \n" % (geological_time/time_scale,EK)) ; EK_file.flush()
    TVD_file.write("%e %e \n" % (geological_time/time_scale,TVD)) ; TVD_file.flush()

    print("     istep= %.6d ; vrms   = %.3e %s" %(istep,vrms/vel_scale,vel_unit))

    print("compute global quantities: %.3f s" % (clock.time()-start)) ; timings[6]+=clock.time()-start

    ###############################################################################################
    #@@ compute nodal heat flux 
    # ordering 0-1-2-3 is BL-BR-TR-TL
    ###############################################################################################
    start=clock.time()

    if solve_T: 
       dTdx_n,dTdz_n,qx_n,qz_n=\
       compute_nodal_heat_flux(icon_V,T,hcond_n,nn_V,m_V,nel,dNdr_V_n,dNdt_V_n,jcbi00n,jcbi01n,jcbi10n,jcbi11n)

       print("     -> dTdx_n (m,M) %.3e %.3e " %(np.min(dTdx_n),np.max(dTdx_n)))
       print("     -> dTdz_n (m,M) %.3e %.3e " %(np.min(dTdz_n),np.max(dTdz_n)))
       print("     -> qx_n (m,M) %.3e %.3e " %(np.min(qx_n),np.max(qx_n)))
       print("     -> qz_n (m,M) %.3e %.3e " %(np.min(qz_n),np.max(qz_n)))

       qx0=qx_n[cornerBL] ; qz0=qz_n[cornerBL]
       qx1=qx_n[cornerBR] ; qz1=qz_n[cornerBR]
       qx2=qx_n[cornerTR] ; qz2=qz_n[cornerTR]
       qx3=qx_n[cornerTL] ; qz3=qz_n[cornerTL]

       corner_q_file.write("%e %e %e %e %e %e %e %e %e\n" % (geological_time/time_scale,\
                                                             qx0,qz0,qx1,qz1,qx2,qz2,qx3,qz3)) 
       corner_q_file.flush()

    else:
       qx_n=0 
       qz_n=0 

    print("compute nodal heat flux: %.3f s" % (clock.time()-start)) ; timings[7]+=clock.time()-start

    ###########################################################################
    #@@ compute heat flux and Nusselt at top and bottom
    ###########################################################################
    start=clock.time()

    if istep%every_Nu==0 and solve_T: 

       avrg_T_bot,avrg_T_top,avrg_dTdz_bot,avrg_dTdz_top,Nu=\
       compute_Nu(Lx,Lz,nel,top_element,bot_element,icon_V,T,dTdz_n,nqperdim,qcoords,qweights,hx)

       print("     -> <T> (bot,top)= %.3e %.3e " %(avrg_T_bot,avrg_T_top))
       print("     -> <dTdz> (bot,top)= %.3e %.3e " %(avrg_dTdz_bot,avrg_dTdz_top))
       print("     -> Nusselt= %.3e " %(Nu))

       Nu_file.write("%e %e \n" % (geological_time/time_scale,Nu)) ; Nu_file.flush()
       avrg_T_bot_file.write("%e %e \n" % (geological_time/time_scale,avrg_T_bot)) ; avrg_T_bot_file.flush()
       avrg_T_top_file.write("%e %e \n" % (geological_time/time_scale,avrg_T_top)) ; avrg_T_top_file.flush()
       avrg_dTdz_bot_file.write("%e %e \n" % (geological_time/time_scale,avrg_dTdz_bot)) ; avrg_dTdz_bot_file.flush()
       avrg_dTdz_top_file.write("%e %e \n" % (geological_time/time_scale,avrg_dTdz_top)) ; avrg_dTdz_top_file.flush()

       print("compute Nu: %.3f s" % (clock.time()-start)) ; timings[8]+=clock.time()-start

    ###########################################################################
    #@@ compute temperature profile
    # not the most elegant but works
    ###########################################################################
    start=clock.time()

    if istep%2500==0 and solve_T: 

       T_profile=np.zeros(nnz,dtype=np.float64)  
       coord_profile=np.zeros(nnz,dtype=np.float64) 

       counter=0    
       for j in range(0,nnz):
           if geometry=='box':
              coord_profile[j]=z_V[counter]
           else:
              coord_profile[j]=rad_V[counter]
           for i in range(0,nnx):
               T_profile[j]+=T[counter]/nnx
               counter+=1
           #end for
       #end for

       np.savetxt('OUTPUT/T_profile_'+str(istep)+'.ascii',np.array([coord_profile,T_profile]).T,header='#z,T')

       print("compute T profile: %.3f s" % (clock.time()-start)) ; timings[9]+=clock.time()-start

    ###########################################################################
    # compute elemental strain rate and deviatoric strainrate
    ###########################################################################
    start=clock.time()

    exx_e,ezz_e,exz_e=compute_elemental_strain_rate(icon_V,u,w,nn_V,nel,x_V,z_V)

    divv_e=exx_e+ezz_e
    dxx_e=exx_e-divv_e/3
    dzz_e=ezz_e-divv_e/3
    dxz_e=exz_e

    print("     -> exx_e (m,M) %.3e %.3e " %(np.min(exx_e),np.max(exx_e)))
    print("     -> ezz_e (m,M) %.3e %.3e " %(np.min(ezz_e),np.max(ezz_e)))
    print("     -> exz_e (m,M) %.3e %.3e " %(np.min(exz_e),np.max(exz_e)))

    if debug_ascii: 
       np.savetxt('DEBUG/strainrate_cartesian_e.ascii',\
                  np.array([x_e,z_e,exx_e,ezz_e,exz_e,effective(exx_e,ezz_e,exz_e)]).T,header='#x,z,exx,ezz,exz,e')

    if geometry=='quarter' or geometry=='half' or geometry=='eighth':    
       if axisymmetric:
          err_e,ett_e,ert_e=convert_tensor_to_spherical_coords(theta_e,exx_e,ezz_e,exz_e)
          drr_e,dtt_e,drt_e=convert_tensor_to_spherical_coords(theta_e,dxx_e,dzz_e,dxz_e)
       else:
          err_e,ett_e,ert_e=convert_tensor_to_polar_coords(theta_e,exx_e,ezz_e,exz_e)
          drr_e,dtt_e,drt_e=convert_tensor_to_polar_coords(theta_e,dxx_e,dzz_e,dxz_e)

       print("     -> err_e (m,M) %.3e %.3e " %(np.min(err_e),np.max(err_e)))
       print("     -> ett_e (m,M) %.3e %.3e " %(np.min(ett_e),np.max(ett_e)))
       print("     -> ert_e (m,M) %.3e %.3e " %(np.min(ert_e),np.max(ert_e)))

       if debug_ascii: np.savetxt('DEBUG/strainrate_polar_e.ascii',np.array([x_e,z_e,err_e,ett_e,ert_e]).T,header='#x,z,err,ett,ert')

       np.savetxt('OUTPUT/top_err_e_'+str(istep)+'.ascii',np.array([theta_e[top_element],err_e[top_element]]).T)
       np.savetxt('OUTPUT/top_drr_e_'+str(istep)+'.ascii',np.array([theta_e[top_element],drr_e[top_element]]).T)

    print("compute elemental sr: %.3f s" % (clock.time()-start)) ; timings[29]+=clock.time()-start

    ###########################################################################
    #@@ compute nodal strainrate
    # method 2 is probably bit more accurate, but more expensive
    ###########################################################################
    start=clock.time()

    if method_nodal_strain_rate==1:
       exx_n,ezz_n,exz_n=compute_nodal_strain_rate(icon_V,u,w,nn_V,m_V,nel,dNdr_V_n,dNdt_V_n,\
                                                   jcbi00n,jcbi01n,jcbi10n,jcbi11n)

    if method_nodal_strain_rate==2: 
       exx_n,ezz_n,exz_n=compute_nodal_strain_rate2(bignb_T,II_T,JJ_T,m_T,nqel,icon_V,u,w,nn_V,nel,JxWq,\
                                                    N_V,dNdr_V,dNdt_V,jcbi00q,jcbi01q,jcbi10q,jcbi11q)

    e_n=effective(exx_n,ezz_n,exz_n)

    print("     -> exx_n (m,M) %.3e %.3e " %(np.min(exx_n),np.max(exx_n)))
    print("     -> ezz_n (m,M) %.3e %.3e " %(np.min(ezz_n),np.max(ezz_n)))
    print("     -> exz_n (m,M) %.3e %.3e " %(np.min(exz_n),np.max(exz_n)))

    if debug_ascii: np.savetxt('DEBUG/strainrate_cartesian_n.ascii',\
                               np.array([x_V,z_V,exx_n,ezz_n,exz_n,e_n,rad_V,theta_V]).T,\
                               header='#x,z,exx,ezz,exz,e,rad,theta')

    if geometry=='quarter' or geometry=='half' or geometry=='eighth':    
       if axisymmetric:
          err_n,ett_n,ert_n=convert_tensor_to_spherical_coords(theta_V,exx_n,ezz_n,exz_n)
          if debug_ascii: np.savetxt('DEBUG/strainrate_spherical_coords.ascii',\
                                     np.array([x_V,z_V,err_n,ett_n,ert_n,rad_V,theta_V]).T,\
                                     header='#x,z,err,ett,ert,rad,theta')
       else:
          err_n,ett_n,ert_n=convert_tensor_to_polar_coords(theta_V,exx_n,ezz_n,exz_n)
          if debug_ascii: np.savetxt('DEBUG/strainrate_polar_coords.ascii',\
                                     np.array([x_V,z_V,err_n,ett_n,ert_n,rad_V,theta_V]).T,\
                                     header='#x,z,err,ett,ert,rad,theta')

       print("     -> err_n (m,M) %.3e %.3e " %(np.min(err_n),np.max(err_n)))
       print("     -> ett_n (m,M) %.3e %.3e " %(np.min(ett_n),np.max(ett_n)))
       print("     -> ert_n (m,M) %.3e %.3e " %(np.min(ert_n),np.max(ert_n)))

       np.savetxt('OUTPUT/top_err_n_'+str(istep)+'.ascii',np.array([theta_V[top_Vnodes],err_n[top_Vnodes]]).T)
       np.savetxt('OUTPUT/bot_err_n_'+str(istep)+'.ascii',np.array([theta_V[bot_Vnodes],err_n[bot_Vnodes]]).T)

    else:
       err_n=0 ; ett_n=0 ; ert_n=0

       np.savetxt('OUTPUT/top_ezz_n'+str(istep)+'.ascii',np.array([x_V[top_Vnodes],ezz_n[top_Vnodes]]).T)
       np.savetxt('OUTPUT/bot_ezz_n'+str(istep)+'.ascii',np.array([x_V[bot_Vnodes],ezz_n[bot_Vnodes]]).T)

    print("compute nodal sr: %.3f s" % (clock.time()-start)) ; timings[11]+=clock.time()-start

    ###########################################################################
    #@@ compute nodal deviatoric strainrate
    ###########################################################################
    start=clock.time()

    divv_n=exx_n+ezz_n

    dxx_n=exx_n-divv_n/3
    dzz_n=ezz_n-divv_n/3
    dxz_n=exz_n

    print("     -> divv_n (m,M) %.3e %.3e " %(np.min(divv_n),np.max(divv_n)))
    print("     -> dxx_n (m,M) %.3e %.3e " %(np.min(dxx_n),np.max(dxx_n)))
    print("     -> dzz_n (m,M) %.3e %.3e " %(np.min(dzz_n),np.max(dzz_n)))
    print("     -> dxz_n (m,M) %.3e %.3e " %(np.min(dxz_n),np.max(dxz_n)))

    print("compute nodal sr: %.3f s" % (clock.time()-start)) ; timings[11]+=clock.time()-start

    ###########################################################################
    #@@ compute deviatoric stress tensor components
    ###########################################################################
    start=clock.time()

    taurr_n=0
    tautt_n=0
    taurt_n=0
    taurr_e=0
    tautt_e=0
    taurt_e=0

    if solve_Stokes:

       tauxx_n=2*eta_n*dxx_n ; tauxx_e=2*eta_e*dxx_e
       tauzz_n=2*eta_n*dzz_n ; tauzz_e=2*eta_e*dzz_e
       tauxz_n=2*eta_n*dxz_n ; tauxz_e=2*eta_e*dxz_e

       if geometry=='box':
          np.savetxt('OUTPUT/top_tauzz_n_'+str(istep)+'.ascii',np.array([x_V[top_Vnodes],tauzz_n[top_Vnodes]]).T)
          np.savetxt('OUTPUT/bot_tauzz_n_'+str(istep)+'.ascii',np.array([x_V[bot_Vnodes],tauzz_n[bot_Vnodes]]).T)

       if geometry=='quarter' or geometry=='half' or geometry=='eighth':
          if axisymmetric:
             taurr_n,tautt_n,taurt_n=convert_tensor_to_spherical_coords(theta_V,tauxx_n,tauzz_n,tauxz_n)
             taurr_e,tautt_e,taurt_e=convert_tensor_to_spherical_coords(theta_e,tauxx_e,tauzz_e,tauxz_e)
          else:
             taurr_n,tautt_n,taurt_n=convert_tensor_to_polar_coords(theta_V,tauxx_n,tauzz_n,tauxz_n)
             taurr_e,tautt_e,taurt_e=convert_tensor_to_polar_coords(theta_e,tauxx_e,tauzz_e,tauxz_e)

          np.savetxt('OUTPUT/top_taurr_n_'+str(istep)+'.ascii',np.array([theta_V[top_Vnodes],taurr_n[top_Vnodes]]).T)
          np.savetxt('OUTPUT/bot_taurr_n_'+str(istep)+'.ascii',np.array([theta_V[bot_Vnodes],taurr_n[bot_Vnodes]]).T)
          np.savetxt('OUTPUT/top_taurr_e'+str(istep)+'.ascii',np.array([theta_e[top_element],taurr_e[top_element]]).T)
          np.savetxt('OUTPUT/bot_taurr_e'+str(istep)+'.ascii',np.array([theta_e[bot_element],taurr_e[bot_element]]).T)

    print("compute deviatoric stress: %.3f s" % (clock.time()-start)) ; timings[27]+=clock.time()-start

    ###########################################################################
    #@@ compute full stress tensor components
    ###########################################################################
    start=clock.time()

    if solve_Stokes:

       sigmaxx_n=-q+tauxx_n ; sigmaxx_e=-p_e+tauxx_e
       sigmazz_n=-q+tauzz_n ; sigmazz_e=-p_e+tauzz_e
       sigmaxz_n=   tauxz_n ; sigmaxz_e=     tauxz_e

       if geometry=='quarter' or geometry=='half' or geometry=='eighth':
          if axisymmetric:
             sigmarr_n,sigmatt_n,sigmart_n=convert_tensor_to_spherical_coords(theta_V,sigmaxx_n,sigmazz_n,sigmaxz_n)
             sigmarr_e,sigmatt_e,sigmart_e=convert_tensor_to_spherical_coords(theta_e,sigmaxx_e,sigmazz_e,sigmaxz_e)
          else:
             sigmarr_n,sigmatt_n,sigmart_n=convert_tensor_to_polar_coords(theta_V,sigmaxx_n,sigmazz_n,sigmaxz_n)
             sigmarr_e,sigmatt_e,sigmart_e=convert_tensor_to_polar_coords(theta_e,sigmaxx_e,sigmazz_e,sigmaxz_e)

       if geometry=='box':
          np.savetxt('OUTPUT/top_sigmazz_n_'+str(istep)+'.ascii',np.array([x_V[top_Vnodes],sigmazz_n[top_Vnodes]]).T)
          np.savetxt('OUTPUT/bot_sigmazz_n_'+str(istep)+'.ascii',np.array([x_V[bot_Vnodes],sigmazz_n[bot_Vnodes]]).T)
          np.savetxt('OUTPUT/top_sigmazz_e_'+str(istep)+'.ascii',np.array([x_e[top_element],sigmazz_e[top_element]]).T)
          np.savetxt('OUTPUT/bot_sigmazz_e_'+str(istep)+'.ascii',np.array([x_e[bot_element],sigmazz_e[bot_element]]).T)

       if geometry=='quarter' or geometry=='half' or geometry=='eighth':
          np.savetxt('OUTPUT/top_sigmarr_n_'+str(istep)+'.ascii',np.array([theta_V[top_Vnodes],sigmarr_n[top_Vnodes]]).T)
          np.savetxt('OUTPUT/bot_sigmarr_n_'+str(istep)+'.ascii',np.array([theta_V[bot_Vnodes],sigmarr_n[bot_Vnodes]]).T)
          np.savetxt('OUTPUT/top_sigmarr_e_'+str(istep)+'.ascii',np.array([theta_e[top_element],sigmarr_e[top_element]]).T)
          np.savetxt('OUTPUT/bot_sigmarr_e_'+str(istep)+'.ascii',np.array([theta_e[bot_element],sigmarr_e[bot_element]]).T)

    else:
       sigmaxx_n=0 ; sigmazz_n=0 ; sigmaxz_n=0
       sigmaxx_e=0 ; sigmazz_e=0 ; sigmaxz_e=0

    print("compute full stress: %.3f s" % (clock.time()-start)) ; timings[27]+=clock.time()-start

    ###########################################################################
    #@@ compute dynamic topography at bottom and surface topo 
    ###########################################################################
    start=clock.time()

    if solve_Stokes:
       if geometry=='box':
          #
          avrg_sigmazz=np.average(sigmazz_n[top_Vnodes])
          dyn_topo_top=(sigmazz_n[top_Vnodes]-avrg_sigmazz)/gz_n[top_Vnodes]/(rho_n[top_Vnodes]-rho_DT_top)
          np.savetxt('OUTPUT/top_dynamic_topography_n_'+str(istep)+'.ascii',np.array([x_V[top_Vnodes],dyn_topo_top]).T)
          #
          avrg_sigmazz=np.average(sigmazz_n[bot_Vnodes])
          dyn_topo_bot=(sigmazz_n[bot_Vnodes]-avrg_sigmazz)/gz_n[bot_Vnodes]/(rho_n[bot_Vnodes]-rho_DT_bot)
          np.savetxt('OUTPUT/bot_dynamic_topography_n_'+str(istep)+'.ascii',np.array([x_V[bot_Vnodes],dyn_topo_bot]).T)

       if geometry=='quarter' or geometry=='half':
          #
          avrg_sigmarr=np.average(sigmarr_n[top_Vnodes])
          dyn_topo_top=(sigmarr_n[top_Vnodes]-avrg_sigmarr)/gr_n[top_Vnodes]/(rho_n[top_Vnodes]-rho_DT_top)
          np.savetxt('OUTPUT/top_dynamic_topography_n_'+str(istep)+'.ascii',np.array([theta_V[top_Vnodes],dyn_topo_top]).T)
          #
          avrg_sigmarr=np.average(sigmarr_e[top_element])
          dyn_topo_top=(sigmarr_e[top_element]-avrg_sigmarr)/gr_e[top_element]/(rho_e[top_element]-rho_DT_top)
          np.savetxt('OUTPUT/top_dynamic_topography_e_'+str(istep)+'.ascii',np.array([theta_e[top_element],dyn_topo_top]).T)
          #
          avrg_sigmarr=np.average(sigmarr_n[bot_Vnodes])
          dyn_topo_bot=(sigmarr_n[bot_Vnodes]-avrg_sigmarr)/gr_n[bot_Vnodes]/(rho_n[bot_Vnodes]-rho_DT_bot)
          np.savetxt('OUTPUT/bot_dynamic_topography_n_'+str(istep)+'.ascii',np.array([theta_V[bot_Vnodes],dyn_topo_bot]).T)
          #
          avrg_sigmarr=np.average(sigmarr_e[bot_element])
          dyn_topo_bot=(sigmarr_e[bot_element]-avrg_sigmarr)/gr_e[bot_element]/(rho_e[bot_element]-rho_DT_bot)
          np.savetxt('OUTPUT/bot_dynamic_topography_e_'+str(istep)+'.ascii',np.array([theta_e[bot_element],dyn_topo_bot]).T)

    print("compute dynamic topo: %.3f s" % (clock.time()-start)) ; timings[26]+=clock.time()-start

    ###########################################################################
    #@@ compute nodal pressure gradient 
    ###########################################################################
    start=clock.time()

    if solve_Stokes:
       dpdx_n,dpdz_n=compute_nodal_pressure_gradient(icon_V,q,nn_V,m_V,nel,dNdr_V_n,dNdt_V_n,\
                                                     jcbi00n,jcbi01n,jcbi10n,jcbi11n)

    print("     -> dpdx_n (m,M) %.3e %.3e " %(np.min(dpdx_n),np.max(dpdx_n)))
    print("     -> dpdz_n (m,M) %.3e %.3e " %(np.min(dpdz_n),np.max(dpdz_n)))

    if debug_ascii: np.savetxt('DEBUG/pressure_gradient.ascii',np.array([x_V,z_V,dpdx_n,dpdz_n]).T,header='#x,z,dpdx,dpdz')

    print("compute nodal pressure gradient: %.3f s" % (clock.time()-start)) ; timings[8]+=clock.time()-start

    ###########################################################################
    #@@ advect particles
    ###########################################################################
    start=clock.time()

    if solve_Stokes:
       match geometry:
        case 'box' :
         swarm_x,swarm_z,swarm_u,swarm_w,swarm_active=\
         advect_particles___box(RKorder,dt,nparticle,swarm_x,swarm_z,swarm_active,\
                                u,w,Lx,Lz,hx,hz,nelx,icon_V,x_V,z_V)
        case 'quarter' :
         swarm_x,swarm_z,swarm_rad,swarm_theta,swarm_u,swarm_w,swarm_active=\
         advect_particles___quarter(RKorder,dt,nparticle,swarm_x,swarm_z,
                                    swarm_rad,swarm_theta,swarm_active,u,w,
                                    Rinner,Router,hrad,htheta,nelx,icon_V,rad_V,theta_V)
        case 'half' :
         swarm_x,swarm_z,swarm_rad,swarm_theta,swarm_u,swarm_w,swarm_active=\
         advect_particles___half(RKorder,dt,nparticle,swarm_x,swarm_z,
                                 swarm_rad,swarm_theta,swarm_active,u,w,
                                 Rinner,Router,hrad,htheta,nelx,icon_V,rad_V,theta_V)
        case 'eighth' :
         swarm_x,swarm_z,swarm_rad,swarm_theta,swarm_u,swarm_w,swarm_active=\
         advect_particles___eighth(RKorder,dt,nparticle,swarm_x,swarm_z,
                                   swarm_rad,swarm_theta,swarm_active,u,w,
                                   Rinner,Router,hrad,htheta,nelx,icon_V,rad_V,theta_V)
        case _ :
         exit('advect_particles not implemented for this geometry')

       if debug_ascii: np.savetxt('DEBUG/swarm.ascii',np.array([swarm_x,swarm_z]).T,header='#x,z')

       print("     -> nb inactive particles:",nparticle-np.sum(swarm_active))
       print("     -> swarm_x (m,M) %.3e %.3e " %(np.min(swarm_x),np.max(swarm_x)))
       print("     -> swarm_z (m,M) %.3e %.3e " %(np.min(swarm_z),np.max(swarm_z)))
       print("     -> swarm_u (m,M) %.3e %.3e " %(np.min(swarm_u),np.max(swarm_u)))
       print("     -> swarm_w (m,M) %.3e %.3e " %(np.min(swarm_w),np.max(swarm_w)))

    else:
       swarm_u=0 ; swarm_w=0 

    print("advect particles: %.3f s" % (clock.time()-start)) ; timings[13]+=clock.time()-start

    ###########################################################################
    #@@ locate particles and compute reduced coordinates
    ###########################################################################
    start=clock.time()

    match geometry:
     case 'box' :
      swarm_r,swarm_t,swarm_iel=\
      locate_particles___box(nparticle,swarm_x,swarm_z,hx,hz,x_V,z_V,icon_V,nelx)
     case 'quarter' | 'half' | 'eighth':
      swarm_r,swarm_t,swarm_iel=\
      locate_particles___annulus(nparticle,swarm_rad,swarm_theta,hrad,htheta,rad_V,theta_V,icon_V,nelx,Rinner)
     case _ :
      exit('locate particles not implemented for ths geometry')

    print("     -> swarm_r (m,M) %e %e " %(np.min(swarm_r),np.max(swarm_r)))
    print("     -> swarm_t (m,M) %e %e " %(np.min(swarm_t),np.max(swarm_t)))
    print("     -> swarm_iel (m,M) %d %d " %(np.min(swarm_iel),np.max(swarm_iel)))
    
    if np.min(swarm_r)<-1 or np.max(swarm_r)>1: exit('r value out of bounds')
    if np.min(swarm_t)<-1 or np.max(swarm_t)>1: exit('t value out of bounds')

    print("locate particles: %.3fs" % (clock.time()-start)) ; timings[16]+=clock.time()-start

    ###########################################################################
    #@@ compute strain on particles
    ###########################################################################

    swarm_strain+=np.sqrt(0.5*(swarm_exx**2+swarm_ezz**2)+swarm_exz**2)*dt

    print("     -> swarm_strain (m,M) %e %e " %(np.min(swarm_strain),np.max(swarm_strain)))

    ###########################################################################
    #@@ output min/max coordinates of each material in one single file
    ###########################################################################
    start=clock.time()

    imat=np.min(swarm_mat)
    jmat=np.max(swarm_mat)
    mats=np.zeros(4*(jmat-imat+1)+1,dtype=np.float64)

    mats[0]=geological_time/time_scale

    counter=1
    for i in range(imat,jmat+1):
        xmin=np.min(swarm_x[swarm_mat==i]) ; mats[counter]=xmin ; counter+=1
        xmax=np.max(swarm_x[swarm_mat==i]) ; mats[counter]=xmax ; counter+=1
        zmin=np.min(swarm_z[swarm_mat==i]) ; mats[counter]=zmin ; counter+=1
        zmax=np.max(swarm_z[swarm_mat==i]) ; mats[counter]=zmax ; counter+=1

    mats.tofile(mats_file,sep=' ',format='%.4e ') ; mats_file.write('\n')
    mats_file.flush()

    print("write min/max extents: %.3fs" % (clock.time()-start)) #; timings[16]+=clock.time()-start

    ###########################################################################
    #@@ generate/write in pvd files
    ###########################################################################

    if istep==0:
       pvd_solution_file.write('<?xml version="1.0"?> \n')
       pvd_solution_file.write('<VTKFile type="Collection" version="0.1" ByteOrder="LittleEndian"> \n')
       pvd_solution_file.write('  <Collection> \n')
       pvd_swarm_file.write('<?xml version="1.0"?> \n')
       pvd_swarm_file.write('<VTKFile type="Collection" version="0.1" ByteOrder="LittleEndian"> \n')
       pvd_swarm_file.write('  <Collection> \n')
    
    if istep%every_solution_vtu==0 or istep==nstep-1: 
       pvd_solution_file.write('    <DataSet timestep="%s" group="" part="0" file="solution_%04d.vtu"/>  \n'\
                             %(geological_time,istep))
       pvd_solution_file.flush()

    if istep%every_swarm_vtu==0 or istep==nstep-1: 
       pvd_swarm_file.write('    <DataSet timestep="%s" group="" part="0" file="swarm_%04d.vtu"/>  \n'\
                             %(geological_time,istep))
       pvd_swarm_file.flush()

    ###########################################################################
    #@@ output solution to vtu file
    ###########################################################################
    start=clock.time()

    if istep%every_solution_vtu==0 or istep==nstep-1: 
       output_solution_to_vtu(solve_Stokes,istep,nel,nn_V,m_V,solve_T,vel_scale,vel_unit,TKelvin,x_V,z_V,\
                              u,w,q,T,eta_n,rho_n,exx_n,ezz_n,exz_n,e_n,divv_n,qx_n,qz_n,rho_e,\
                              sigmaxx_n,sigmazz_n,sigmaxz_n,rad_V,theta_V,eta_e,nparticle_e,area,icon_V,\
                              bc_fix_V,bc_fix_T,geometry,gx_n,gz_n,err_n,ett_n,ert_n,vr,vt,plith,\
                              exx_e,ezz_e,exz_e,taurr_n,tautt_n,taurt_n)

       print("output solution to vtu file: %.3f s" % (clock.time()-start)) ; timings[10]+=clock.time()-start

    ########################################################################
    #@@ output particles to vtu file
    ########################################################################
    start=clock.time()

    if istep%every_swarm_vtu==0 or istep==nstep-1: 
       output_swarm_to_vtu(solve_Stokes,use_melting,TKelvin,istep,geometry,nparticle,solve_T,vel_scale,swarm_x,swarm_z,\
                           swarm_u,swarm_w,swarm_mat,swarm_rho,swarm_eta,swarm_r,swarm_t,swarm_p,\
                           swarm_paint,swarm_exx,swarm_ezz,swarm_exz,swarm_T,swarm_iel,\
                           swarm_hcond,swarm_hcapa,swarm_rad,swarm_theta,swarm_strain,swarm_F,swarm_sst) 

       print("output particles to vtu file: %.3f s" % (clock.time()-start)) ; timings[20]+=clock.time()-start

    ########################################################################
    #@@ output quadrature points to vtu file
    ########################################################################
    start=clock.time()

    if istep%every_quadpoints_vtu==0 or istep==nstep-1: 
       output_quadpoints_to_vtu(istep,nel,nqel,nq,solve_T,xq,zq,rhoq,etaq,Tq,hcondq,hcapaq,dpdxq,dpdzq,gxq,gzq)

       print("output quad pts to vtu file: %.3f s" % (clock.time()-start)) ; timings[22]+=clock.time()-start

    ########################################################################
    #@@ compute gravitational field above domain 
    # xs[npts],ys: coordinates of satellite
    # gxI,gzI,gnormI: gravity from internal density distribution
    # gxDTt,gzDTt,gnormDTt: gravity from dynamic topography at top 
    # gxDTb,gzDTb,gnormDTb: gravity from dynamic topography at bottom
    ########################################################################
    start=clock.time()

    if gravity_npts>0:

       if istep==0:
          xs=np.zeros(gravity_npts,dtype=np.float64)  
          zs=np.zeros(gravity_npts,dtype=np.float64)  
          gxI=np.zeros((gravity_npts,nstep),dtype=np.float64)  
          gzI=np.zeros((gravity_npts,nstep),dtype=np.float64)  
          grI=np.zeros((gravity_npts,nstep),dtype=np.float64)  
          gtI=np.zeros((gravity_npts,nstep),dtype=np.float64)  
          gnormI=np.zeros((gravity_npts,nstep),dtype=np.float64)  
          gnormI_rate=np.zeros((gravity_npts,nstep),dtype=np.float64)  
          gxDTt=np.zeros((gravity_npts,nstep),dtype=np.float64)  
          gzDTt=np.zeros((gravity_npts,nstep),dtype=np.float64)  
          gnormDTt=np.zeros((gravity_npts,nstep),dtype=np.float64)  
          gnormDTt_rate=np.zeros((gravity_npts,nstep),dtype=np.float64)  
          gxDTb=np.zeros((gravity_npts,nstep),dtype=np.float64)  
          gzDTb=np.zeros((gravity_npts,nstep),dtype=np.float64)  
          gnormDTb=np.zeros((gravity_npts,nstep),dtype=np.float64)  
          gnormDTb_rate=np.zeros((gravity_npts,nstep),dtype=np.float64)  

       match(geometry):
        case 'box' :
         for i in range(0,gravity_npts):
             xs[i]=i*Lx/(gravity_npts-1)
             zs[i]=Lz+gravity_height
             gxI[i,istep],gzI[i,istep],gnormI[i,istep]=\
             compute_gravity_at_point(xs[i],zs[i],nel,x_e,z_e,rho_e,area,gravity_rho_ref)

             gxDTt[i,istep],gzDTt[i,istep],gnormDTt[i,istep]=\
             compute_gravity_fromDT_at_point(xs[i],zs[i],Lz,nelx,x_V[top_Vnodes],\
                                             rho_n[top_Vnodes],dyn_topo_top,rho_DT_top)

             gxDTb[i,istep],gzDTb[i,istep],gnormDTb[i,istep]=\
             compute_gravity_fromDT_at_point(xs[i],zs[i],0,nelx,x_V[bot_Vnodes],\
                                                rho_n[bot_Vnodes],dyn_topo_bot,rho_DT_bot)

         np.savetxt('OUTPUT/gravityI_'+str(istep)+'.ascii',\
                    np.array([xs,zs,gnormI[:,istep],gxI[:,istep],gzI[:,istep]]).T,header='#x,z,g,gx,gz')
         np.savetxt('OUTPUT/gravityDTt_'+str(istep)+'.ascii',\
                    np.array([xs,zs,gnormDTt[:,istep],gxDTt[:,istep],gzDTt[:,istep]]).T,header='#x,z,g,gx,gz')
         np.savetxt('OUTPUT/gravityDTb_'+str(istep)+'.ascii',\
                    np.array([xs,zs,gnormDTb[:,istep],gxDTb[:,istep],gzDTb[:,istep]]).T,header='#x,z,g,gx,gz')

        case 'quarter' | 'half':
         for i in range(0,gravity_npts):
             xs[i]=(Router+gravity_height)*np.cos(i/(gravity_npts-1)*opening_angle+theta_min)
             zs[i]=(Router+gravity_height)*np.sin(i/(gravity_npts-1)*opening_angle+theta_min)
             gxI[i,istep],gzI[i,istep],gnormI[i,istep]=\
             compute_gravity_at_point(xs[i],zs[i],nel,x_e,z_e,rho_e,area,gravity_rho_ref)

         rads=np.sqrt(xs**2+zs**2)
         thetas=np.pi/2-np.arctan2(xs,zs)
         grI[:,istep]=gxI[:,istep]*np.cos(thetas)+gzI[:,istep]*np.sin(thetas)
         gtI[:,istep]=-gxI[:,istep]*np.sin(thetas)+gzI[:,istep]*np.cos(thetas)

         np.savetxt('OUTPUT/gravityI_'+str(istep)+'.ascii',\
                    np.array([rads,thetas,gnormI[:,istep],grI[:,istep],gtI[:,istep]]).T,header='#r,theta,g,gx,gz')

        case _ :
         print('gravity calculations not available for this geometry')   

       if istep>0:
          gnormI_rate[:,istep]=(gnormI[:,istep]-gnormI[:,istep-1])/dt
          gnormDTt_rate[:,istep]=(gnormDTt[:,istep]-gnormDTt[:,istep-1])/dt
          gnormDTb_rate[:,istep]=(gnormDTb[:,istep]-gnormDTb[:,istep-1])/dt
          if geometry=='box':
             np.savetxt('OUTPUT/gravityI_rate_'+str(istep)+'.ascii',np.array([xs,gnormI_rate[:,istep]/mGal*year]).T,header='#x,g')
             np.savetxt('OUTPUT/gravityDTt_rate_'+str(istep)+'.ascii',np.array([xs,gnormDTt_rate[:,istep]/mGal*year]).T,header='#x,g')
             np.savetxt('OUTPUT/gravityDTb_rate_'+str(istep)+'.ascii',np.array([xs,gnormDTb_rate[:,istep]/mGal*year]).T,header='#x,g')
          if geometry=='quarter' or geometry=='half':
             np.savetxt('OUTPUT/gravityI_rate_'+str(istep)+'.ascii',np.array([thetas,gnormI_rate[:,istep]/mGal*year]).T,header='#theta,g')

    print("compute gravity: %.3f s" % (clock.time()-start)) ; timings[23]+=clock.time()-start

    ###########################################################################
    #@@ assess steady state
    ###########################################################################
    start=clock.time()

    if solve_Stokes:
       steady_state_u=np.linalg.norm(u_mem-u,2)/np.linalg.norm(u,2)<tol_ss
       steady_state_w=np.linalg.norm(w_mem-w,2)/np.linalg.norm(w,2)<tol_ss
       steady_state_p=np.linalg.norm(p_mem-p,2)/np.linalg.norm(p,2)<tol_ss
    else:
       steady_state_u=False
       steady_state_w=False
       steady_state_p=False

    if solve_T:
       steady_state_T=np.linalg.norm(T_mem-T,2)/np.linalg.norm(T,2)<tol_ss
       print('     -> steady state u,w,p,T',steady_state_u,steady_state_w,\
                                            steady_state_p,steady_state_T)
    else:
       steady_state_T=True
       print('     -> steady state u,w,p',steady_state_u,steady_state_w,\
                                          steady_state_p)

    steady_state=steady_state_u and steady_state_w and\
                 steady_state_p and steady_state_T 

    u_mem=u.copy()
    w_mem=w.copy()
    p_mem=p.copy()
    if solve_T: T_mem=T.copy()

    print("assess steady state: %.4f s" % (clock.time()-start)) #; timings[22]+=clock.time()-start

    ###########################################################################

    if istep%10==0 or istep==nstep-1 or geological_time>end_time:

       duration=clock.time()-topstart

       print("----------------------------------------------------------------------")
       print("build FE matrix V: %8.3f s      (%.3f s per call) | %5.2f percent" % (timings[1],timings[1]/(istep+1),timings[1]/duration*100)) 
       print("solve system V: %8.3f s         (%.3f s per call) | %5.2f percent" % (timings[2],timings[2]/(istep+1),timings[2]/duration*100))
       print("build matrix T: %8.3f s         (%.3f s per call) | %5.2f percent" % (timings[4],timings[4]/(istep+1),timings[4]/duration*100))
       print("solve system T: %8.3f s         (%.3f s per call) | %5.2f percent" % (timings[5],timings[5]/(istep+1),timings[5]/duration*100))
       print("compute plith: %8.3f s          (%.3f s per call) | %5.2f percent" % (timings[28],timings[28]/(istep+1),timings[28]/duration*100))
       print("comp. glob quantities: %8.3f s  (%.3f s per call) | %5.2f percent" % (timings[6],timings[6]/(istep+1),timings[6]/duration*100))
       print("comp. nodal p: %8.3f s          (%.3f s per call) | %5.2f percent" % (timings[3],timings[3]/(istep+1),timings[3]/duration*100))
       print("comp. nodal sr: %8.3f s         (%.3f s per call) | %5.2f percent" % (timings[11],timings[11]/(istep+1),timings[11]/duration*100))
       print("comp. nodal stress: %8.3f s     (%.3f s per call) | %5.2f percent" % (timings[27],timings[27]/(istep+1),timings[27]/duration*100))
       print("comp. nodal heat flux: %8.3f s  (%.3f s per call) | %5.2f percent" % (timings[7],timings[7]/(istep+1),timings[7]/duration*100))
       print("comp. nodal press grad: %8.3f s (%.3f s per call) | %5.2f percent" % (timings[8],timings[8]/(istep+1),timings[8]/duration*100))
       print("comp. eltal sr: %8.3f s         (%.3f s per call) | %5.2f percent" % (timings[29],timings[29]/(istep+1),timings[29]/duration*100))
       print("comp. T profile: %8.3f s        (%.3f s per call) | %5.2f percent" % (timings[9],timings[9]/(istep+1),timings[9]/duration*100)) 
       print("normalise pressure: %8.3f s     (%.3f s per call) | %5.2f percent" % (timings[12],timings[12]/(istep+1),timings[12]/duration*100))
       print("advect particles: %8.3f s       (%.3f s per call) | %5.2f percent" % (timings[13],timings[13]/(istep+1),timings[13]/duration*100))
       print("split solution: %8.3f s         (%.3f s per call) | %5.2f percent" % (timings[14],timings[14]/(istep+1),timings[14]/duration*100))
       print("material model on ptcls: %8.3fs (%.3f s per call) | %5.2f percent" % (timings[15],timings[15]/(istep+1),timings[15]/duration*100))
       print("locate particles: %8.3f s       (%.3f s per call) | %5.2f percent" % (timings[16],timings[16]/(istep+1),timings[16]/duration*100))
       print("comp eltal rho,eta: %8.3f s     (%.3f s per call) | %5.2f percent" % (timings[17],timings[17]/(istep+1),timings[17]/duration*100))
       print("comp nodal rho,eta: %8.3f s     (%.3f s per call) | %5.2f percent" % (timings[18],timings[18]/(istep+1),timings[18]/duration*100))
       print("compute dyn topo: %8.3f s       (%.3f s per call) | %5.2f percent" % (timings[26],timings[26]/(istep+1),timings[26]/duration*100))
       print("comp timestep: %8.3f s          (%.3f s per call) | %5.2f percent" % (timings[19],timings[19]/(istep+1),timings[19]/duration*100))
       print("output solution to vtu: %8.3f s (%.3f s per call) | %5.2f percent" % (timings[10],timings[10]/(istep+1),timings[10]/duration*100))
       print("output swarm to vtu: %8.3f s    (%.3f s per call) | %5.2f percent" % (timings[20],timings[20]/(istep+1),timings[20]/duration*100))
       print("output qpts to vtu: %8.3f s     (%.3f s per call) | %5.2f percent" % (timings[22],timings[22]/(istep+1),timings[22]/duration*100))
       print("project fields on qpts: %8.3f s (%.3f s per call) | %5.2f percent" % (timings[21],timings[21]/(istep+1),timings[21]/duration*100))
       print("compute gravity: %8.3f s        (%.3f s per call) | %5.2f percent" % (timings[23],timings[23]/(istep+1),timings[23]/duration*100))
       print("interp sr,p,T on ptcls: %8.3f s (%.3f s per call) | %5.2f percent" % (timings[24],timings[24]/(istep+1),timings[24]/duration*100))
       #print("interp T on ptcls: %8.3f s      (%.3f s per call) | %5.2f percent" % (timings[25],timings[25]/(istep+1),timings[25]/duration*100))
       print("----------------------------------------------------------------------")
       print("compute time per timestep: %.2f" %(duration/(istep+1)))
       print("----------------------------------------------------------------------")

    dtimings=timings-timings_mem
    dtimings[0]=istep ; dtimings.tofile(timings_file,sep=' ',format='%e') ; timings_file.write(" \n" ) 
    timings_mem[:]=timings[:]

    ###########################################################################

    if geometry=='box' and nsamplepoints>0:
       sample_solution_box(nn_V,x_V,z_V,u,w,q,T,nsamplepoints,xsamplepoints,zsamplepoints,Lx,Lz,nelx,nelz)

    ###########################################################################

    if geological_time>end_time: 
       print('***** end time reached *****')
       break

    if steady_state: 
       print('***** steady state reached *****')
       break

#end for istep
#@@ --------------------- end time stepping loop ------------------------------

pvd_solution_file.write('  </Collection>\n')
pvd_solution_file.write('</VTKFile>\n')
pvd_swarm_file.write('  </Collection>\n')
pvd_swarm_file.write('</VTKFile>\n')

###############################################################################
# output horizontal and vertical profiles
###############################################################################
start=clock.time()

np.savetxt('OUTPUT/profile_vertical.ascii',np.array([z_V[middleV_nodes],\
                                                     u[middleV_nodes],\
                                                     w[middleV_nodes],\
                                                     q[middleV_nodes],\
                                                     T[middleV_nodes]]).T)

np.savetxt('OUTPUT/profile_horizontal.ascii',np.array([x_V[middleH_nodes],\
                                                       u[middleH_nodes],\
                                                       w[middleH_nodes],\
                                                       q[middleH_nodes],\
                                                       T[middleH_nodes]]).T)

###############################################################################
# close files
###############################################################################
       
vstats_file.close()
pstats_file.close()
vrms_file.close()
dt_file.close()
TM_file.close()
EK_file.close()
TVD_file.close()
if solve_T:
   Nu_file.close()
   corner_q_file.close()
   Tstats_file.close()
   avrg_T_bot_file.close()
   avrg_T_top_file.close()
   avrg_dTdz_bot_file.close()
   avrg_dTdz_top_file.close()

###############################################################################

print("-----------------------------")
print("total compute time: %.1f s" % (duration))
print("sum timings: %.1f s" % (np.sum(timings)))
print("-----------------------------")
    
###############################################################################
