import numpy as np
import sys as sys
import numba
import random
import time as clock
import scipy.sparse as sps
from constants import *
from scipy import sparse
from PoissonDisc import *
from compute_vrms import *
from pic_functions import *
from basis_functions import *
from build_matrix_stokes import *
from build_matrix_energy import *
from compute_nodal_strain_rate import *
from compute_nodal_heat_flux import *
from compute_nodal_pressure import *
from export_swarm_to_vtu import *
from export_solution_to_vtu import *
from export_quadpoints_to_vtu import *
from compute_gravity_at_point import *
from compute_gravity_fromDT_at_point import *
from project_nodal_field_onto_qpoints import *
from compute_nodal_pressure_gradient import *
from postprocessors import *

print("-----------------------------")
print("----------- MEEUUW ----------")
print("-----------------------------")

###############################################################################
# experiment 0: Blankenbach et al, 1993    - isoviscous convection
# experiment 1: van Keken et al, JGR, 1997 - Rayleigh-Taylor experiment
# experiment 2: Schmeling et al, PEPI 2008 - Newtonian subduction
# experiment 3: Tosi et al, 2015           - visco-plastic convection
# experiment 4: not sure. mantle size convection
# experiment 5: Trompert & Hansen, Nature 1998 - convection w/ plate-like  
# experiment 6: Crameri et al, GJI 2012 (cosine perturbation & plume) 
# experiment 7: ESA workshop
###############################################################################

experiment=7

if int(len(sys.argv)==5):
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
     case _ : exit('setup - unknown experiment')  

if int(len(sys.argv)==5): # override these parameters
   nelx  = int(sys.argv[2])
   nely  = int(sys.argv[3])
   nstep = int(sys.argv[4])

###############################################################################

ndim=2                     # number of dimensions
ndof_V=2                   # number of velocity dofs per node
nel=nelx*nely              # total number of elements
nn_V=(2*nelx+1)*(2*nely+1) # number of V nodes
nn_P=(nelx+1)*(nely+1)     # number of P nodes

m_V=9 # number of velocity nodes per element
m_P=4 # number of pressure nodes per element
m_T=9 # number of temperature nodes per element

r_V=[-1, 1,1,-1, 0,1,0,-1,0]
s_V=[-1,-1,1, 1,-1,0,1, 0,0]

ndof_V_el=m_V*ndof_V

Nfem_V=nn_V*ndof_V # number of velocity dofs
Nfem_P=nn_P        # number of pressure dofs
Nfem_T=nn_V        # number of temperature dofs
Nfem=Nfem_V+Nfem_P # total nb of dofs

hx=Lx/nelx # element size in x direction
hy=Ly/nely # element size in y direction

nparticle_per_element=nparticle_per_dim**2
nparticle=nel*nparticle_per_element

timings=np.zeros(25+1)
timings_mem=np.zeros(25+1)

L_ref=(Lx+Ly)/2

tol_ss=1e-4

###############################################################################
# quadrature rule points and weights
###############################################################################

nqperdim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]
nqel=nqperdim**ndim
nq=nqel*nel

###############################################################################
# open output files & write headers
###############################################################################

vrms_file=open('vrms.ascii',"w") ; vrms_file.write("#time,vrms\n")
pstats_file=open('pressure_stats.ascii',"w") ; pstats_file.write("#istep,min p, max p\n")
vstats_file=open('velocity_stats.ascii',"w") 
vstats_file.write("#istep,min(u),max(u),min(v),max(v)\n")
vstats_file.write("# "+vel_unit+"\n")
Tstats_file=open('temperature_stats.ascii',"w") 
dt_file=open('dt.ascii',"w") ; dt_file.write("#time dt1 dt2 dt\n") ; dt_file.write('#'+time_unit+'\n')
ptcl_stats_file=open('particle_stats.ascii',"w")
Nu_file=open('Nu.ascii',"w") ; Nu_file.write("#time Nu\n")
avrg_T_bot_file=open('avrg_T_bot.ascii',"w") 
avrg_T_top_file=open('avrg_T_top.ascii',"w") 
avrg_dTdy_bot_file=open('avrg_dTdy_bot.ascii',"w") 
avrg_dTdy_top_file=open('avrg_dTdy_top.ascii',"w") 
timings_file=open('timings.ascii',"w")
TM_file=open('total_mass.ascii',"w") 
EK_file=open('kinetic_energy.ascii',"w") 
TVD_file=open('viscous_dissipation.ascii',"w") 
pvd_solution_file=open('solution.pvd',"w")
pvd_swarm_file=open('swarm.pvd',"w")
corner_q_file=open('corner_heat_flux.ascii','w')
mats_file=open('mats.ascii','w')

###############################################################################

print('experiment=',experiment)
print('nelx,nely=',nelx,nely)
print('Lx,Ly=',Lx,Ly)
print('hx,hy=',hx,hy)
print('nn_V=',nn_V)
print('nn_P=',nn_P)
print('nel=',nel)
print('Nfem_V=',Nfem_V)
print('Nfem_P=',Nfem_P)
print('Nfem=',Nfem)
print('nqperdim=',nqperdim)
print('CFLnb=',CFLnb)
print('debug_ascii:',debug_ascii)
print('debug_nan:',debug_nan)
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
print('-----------------------------')

###############################################################################
# build velocity nodes coordinates 
# BL: bottom left, BR: bottom right, TL: top left, TR: top right
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64) 
y_V=np.zeros(nn_V,dtype=np.float64)
top_nodes=np.zeros(nn_V,dtype=bool)
bot_nodes=np.zeros(nn_V,dtype=bool)
left_nodes=np.zeros(nn_V,dtype=bool)
right_nodes=np.zeros(nn_V,dtype=bool)
middleH_nodes=np.zeros(nn_V,dtype=bool)
middleV_nodes=np.zeros(nn_V,dtype=bool)

nnx=2*nelx+1 
nny=2*nely+1 

counter=0    
for j in range(0,2*nely+1):
    for i in range(0,2*nelx+1):
        x_V[counter]=i*hx/2
        y_V[counter]=j*hy/2
        if (i==0): left_nodes[counter]=True
        if (i==2*nelx): right_nodes[counter]=True
        if (j==0): bot_nodes[counter]=True
        if (j==2*nely): top_nodes[counter]=True
        if abs(x_V[counter]/Lx-0.5)<eps: middleV_nodes[counter]=True
        if abs(y_V[counter]/Ly-0.5)<eps: middleH_nodes[counter]=True
        if i==0 and j==0: cornerBL=counter
        if i==nnx-1 and j==0: cornerBR=counter
        if i==0 and j==nny-1: cornerTL=counter
        if i==nnx-1 and j==nny-1: cornerTR=counter
        counter+=1
    #end for
#end for

if debug_ascii: np.savetxt('gridV.ascii',np.array([x_V,y_V]).T,header='# x,y')

print("build V grid: %.3f s" % (clock.time() - start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)
top_element=np.zeros(nel,dtype=bool)
bot_element=np.zeros(nel,dtype=bool)
left_element=np.zeros(nel,dtype=bool)
right_element=np.zeros(nel,dtype=bool)

counter=0
for j in range(0,nely):
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
        if (j==nely-1): top_element[counter]=True
        counter+=1
    #end for
#end for

print("build icon_V: %.3f s" % (clock.time()-start))

###############################################################################
# build pressure grid 
###############################################################################
start=clock.time()

x_P=np.zeros(nn_P,dtype=np.float64)
y_P=np.zeros(nn_P,dtype=np.float64)

counter=0    
for j in range(0,nely+1):
    for i in range(0,nelx+1):
        x_P[counter]=i*hx
        y_P[counter]=j*hy
        counter+=1
    #end for
 #end for

if debug_ascii: np.savetxt('gridP.ascii',np.array([x_P,y_P]).T,header='# x,y')

print("build P grid: %.3f s" % (clock.time() - start))

###############################################################################
# build pressure connectivity array 
###############################################################################
start=clock.time()

icon_P=np.zeros((m_P,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
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
# define velocity boundary conditions
# bc_fix_V is a vector of booleans of size Nfem_V
# bc_val_V is a vector of float64 of size Nfem_V
###############################################################################
start=clock.time()

bc_fix_V,bc_val_V=assign_boundary_conditions_V(x_V,y_V,ndof_V,Nfem_V,nn_V)

print("velocity b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# define temperature boundary conditions
# bc_fix_T is a vector of booleans of size Nfem_T
# bc_val_T is a vector of float64 of size Nfem_T
###############################################################################
start=clock.time()

if solve_T:

   bc_fix_T,bc_val_T=assign_boundary_conditions_T(x_V,y_V,Nfem_T,nn_V)

   print("temperature b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# initial temperature. T is a vector of float64 of size nn_V
# Even if solve_T=False it needs to be allocated
###############################################################################
start=clock.time()

if solve_T: 

   T=initial_temperature(x_V,y_V,nn_V)

   T_mem=T.copy()

   if debug_ascii: np.savetxt('T_init.ascii',np.array([x_V,y_V,T]).T,header='# x,y,T')

   print("     -> T init (m,M) %.3e %.3e " %(np.min(T),np.max(T)))

   print("initial temperature: %.3f s" % (clock.time()-start))

else:

   T=np.zeros(nn_V,dtype=np.float64) 

###############################################################################
# compute area of elements / sanity check
###############################################################################
start=clock.time()

xc=np.zeros(nel,dtype=np.float64) 
yc=np.zeros(nel,dtype=np.float64) 
area=np.zeros(nel,dtype=np.float64) 
jcb=np.zeros((ndim,ndim),dtype=np.float64)

for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            area[iel]+=JxWq
        #end for
    #end for
    xc[iel]=0.5*(x_V[icon_V[0,iel]]+x_V[icon_V[2,iel]])
    yc[iel]=0.5*(y_V[icon_V[0,iel]]+y_V[icon_V[2,iel]])
#end for

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %e %e " %(area.sum(),Lx*Ly))

print("compute elements areas: %.3f s" % (clock.time() - start))

###############################################################################
# Compute jacobian matrix (inverse and determinant)
# This is only valid for rectangular elements!
###############################################################################

jcbi=np.zeros((ndim,ndim),dtype=np.float64)
jcbi[0,0]=2/hx
jcbi[1,1]=2/hy
jcob=hx*hy/4

###############################################################################
# precompute basis functions values at quadrature points
###############################################################################
start=clock.time()

rq=np.zeros(nqel,dtype=np.float64) 
sq=np.zeros(nqel,dtype=np.float64) 
weightq=np.zeros(nqel,dtype=np.float64) 
JxWq=np.zeros(nqel,dtype=np.float64) 
N_V=np.zeros((nqel,m_V),dtype=np.float64) 
N_P=np.zeros((nqel,m_P),dtype=np.float64) 
dNdr_V=np.zeros((nqel,m_V),dtype=np.float64) 
dNds_V=np.zeros((nqel,m_V),dtype=np.float64) 
dNdx_V=np.zeros((nqel,m_V),dtype=np.float64) 
dNdy_V=np.zeros((nqel,m_V),dtype=np.float64) 
   
counterq=0 
for iq in range(0,nqperdim):
    for jq in range(0,nqperdim):
        rq[counterq]=qcoords[iq]
        sq[counterq]=qcoords[jq]
        weightq[counterq]=qweights[iq]*qweights[jq]
        JxWq[counterq]=jcob*weightq[counterq]
        N_V[counterq,0:m_V]=basis_functions_V(rq[counterq],sq[counterq])
        N_P[counterq,0:m_P]=basis_functions_P(rq[counterq],sq[counterq])
        dNdr_V[counterq,0:m_V]=basis_functions_V_dr(rq[counterq],sq[counterq])
        dNds_V[counterq,0:m_V]=basis_functions_V_ds(rq[counterq],sq[counterq])
        dNdx_V[counterq,0:m_V]=jcbi[0,0]*dNdr_V[counterq,0:m_V]
        dNdy_V[counterq,0:m_V]=jcbi[1,1]*dNds_V[counterq,0:m_V]
        counterq+=1
    #end for
#end for

print("compute N & grad(N) at q pts: %.3f s" % (clock.time()-start))

###############################################################################
# precompute basis functions values at V nodes
###############################################################################
start=clock.time()

N_P_n=np.zeros((m_V,m_P),dtype=np.float64) 
dNdx_V_n=np.zeros((m_V,m_V),dtype=np.float64) 
dNdy_V_n=np.zeros((m_V,m_V),dtype=np.float64) 
   
for i in range(0,m_V):
    N_P_n[i,0:m_P]=basis_functions_P(r_V[i],s_V[i])
    dNdx_V_n[i,0:m_V]=jcbi[0,0]*basis_functions_V_dr(r_V[i],s_V[i])
    dNdy_V_n[i,0:m_V]=jcbi[1,1]*basis_functions_V_ds(r_V[i],s_V[i])

print("compute N & grad(N) at V nodes: %.3f s" % (clock.time()-start))

###############################################################################
# compute coordinates of quadrature points
###############################################################################
start=clock.time()

xq=Q2_project_nodal_field_onto_qpoints(x_V,nqel,nel,N_V,icon_V)
yq=Q2_project_nodal_field_onto_qpoints(y_V,nqel,nel,N_V,icon_V)

print("     -> xq (m,M) %.3e %.3e " %(np.min(xq),np.max(xq)))
print("     -> yq (m,M) %.3e %.3e " %(np.min(yq),np.max(yq)))

print("compute coords quad pts: %.3f s" % (clock.time()-start))

###############################################################################
# compute array for assembly
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
# fill I,J arrays for Stokes matrix
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
# fill I,J arrays for temperature matrix
###############################################################################
start=clock.time()

if solve_T:

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
# particle coordinates setup
###############################################################################
start=clock.time()

swarm_x=np.zeros(nparticle,dtype=np.float64)
swarm_y=np.zeros(nparticle,dtype=np.float64)
swarm_r=np.zeros(nparticle,dtype=np.float64)
swarm_s=np.zeros(nparticle,dtype=np.float64)
swarm_iel=np.zeros(nparticle,dtype=np.int32)

match(particle_distribution):

     case(0): # random
         counter=0
         for iel in range(0,nel):
             for im in range(0,nparticle_per_element):
                 r=random.uniform(-1.,+1)
                 s=random.uniform(-1.,+1)
                 N=basis_functions_V(r,s)
                 swarm_x[counter]=np.dot(N[:],x_V[icon_V[:,iel]])
                 swarm_y[counter]=np.dot(N[:],y_V[icon_V[:,iel]])
                 swarm_r[counter]=r
                 swarm_r[counter]=s
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
                     s=-1.+j*2./nparticle_per_dim + 1./nparticle_per_dim
                     N=basis_functions_V(r,s)
                     swarm_x[counter]=np.dot(N[:],x_V[icon_V[:,iel]])
                     swarm_y[counter]=np.dot(N[:],y_V[icon_V[:,iel]])
                     swarm_r[counter]=r
                     swarm_r[counter]=s
                     swarm_iel[counter]=iel
                     counter+=1
                 #end for
             #end for
         #end for

     case(2): # Poisson Disc

         kpoisson=30
         nparticle_wish=nel*nparticle_per_element # target
         print ('     -> nparticle_wish: %d ' % (nparticle_wish) )
         avrgdist=np.sqrt(Lx*Ly/nparticle_wish)/1.25
         nparticle,swarm_x,swarm_y=PoissonDisc(kpoisson,avrgdist,Lx,Ly)
         print ('     -> nparticle: %d ' % (nparticle) )

         swarm_r,swarm_s,swarm_iel=\
         locate_particles(nparticle,swarm_x,swarm_y,hx,hy,x_V,y_V,icon_V,nelx)

     case 3 : # pseudo-random

         counter=0
         for iel in range(0,nel):
             for j in range(0,nparticle_per_dim):
                 for i in range(0,nparticle_per_dim):
                     r=-1.+i*2./nparticle_per_dim + 1./nparticle_per_dim
                     s=-1.+j*2./nparticle_per_dim + 1./nparticle_per_dim
                     r+=random.uniform(-0.2,+0.2)*(2/nparticle_per_dim)
                     s+=random.uniform(-0.2,+0.2)*(2/nparticle_per_dim)
                     N=basis_functions_V(r,s)
                     swarm_x[counter]=np.dot(N[:],x_V[icon_V[:,iel]])
                     swarm_y[counter]=np.dot(N[:],y_V[icon_V[:,iel]])
                     swarm_r[counter]=r
                     swarm_r[counter]=s
                     swarm_iel[counter]=iel
                     counter+=1
                 #end for
             #end for
         #end for

     case _ :
         exit('unknown particle_distribution')

if debug_ascii: np.savetxt('swarm_distribution.ascii',np.array([swarm_x,swarm_y]).T)

swarm_active=np.zeros(nparticle,dtype=bool) ; swarm_active[:]=True

print("     -> nparticle %d " % nparticle)
print("     -> swarm_x (m,M) %.3e %.3e " %(np.min(swarm_x),np.max(swarm_x)))
print("     -> swarm_y (m,M) %.3e %.3e " %(np.min(swarm_y),np.max(swarm_y)))

print("particles setup: %.3f s" % (clock.time()-start))

###############################################################################
# particle paint
###############################################################################
start=clock.time()

swarm_paint=np.zeros(nparticle,dtype=np.int32)

for i in [0,2,4,6,8,10,12,14]:
    dx=Lx/16
    for im in range (0,nparticle):
        if swarm_x[im]>i*dx and swarm_x[im]<(i+1)*dx:
           swarm_paint[im]+=1

for i in [0,2,4,6,8,10,12,14]:
    dy=Ly/16
    for im in range (0,nparticle):
        if swarm_y[im]>i*dy and swarm_y[im]<(i+1)*dy:
           swarm_paint[im]+=1

print("particles paint: %.3f s" % (clock.time()-start))

###############################################################################
# particle layout
###############################################################################
start=clock.time()

swarm_mat=particle_layout(nparticle,swarm_x,swarm_y,Lx,Ly)

print("     -> swarm_mat (m,M) %d %d " %(np.min(swarm_mat),np.max(swarm_mat)))
    
if debug_ascii: 
   np.savetxt('swarm_mat.ascii',np.array([swarm_x,swarm_y,swarm_mat]).T,header='# x,y,mat')

print("particle layout: %.3f s" % (clock.time()-start))

###############################################################################
###############################################################################
###############################################################################
# time stepping loop
###############################################################################
###############################################################################
###############################################################################
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

geological_time=0.
dt1_mem=1e50
dt2_mem=1e50
       
exx_nodal=np.zeros(nn_V,dtype=np.float64)  
eyy_nodal=np.zeros(nn_V,dtype=np.float64)  
exy_nodal=np.zeros(nn_V,dtype=np.float64)  
dpdx_nodal=np.zeros(nn_V,dtype=np.float64)  
dpdy_nodal=np.zeros(nn_V,dtype=np.float64)  
u_mem=np.zeros(nn_V,dtype=np.float64)  
v_mem=np.zeros(nn_V,dtype=np.float64)  
p_mem=np.zeros(nn_P,dtype=np.float64)  

topstart=clock.time()

for istep in range(0,nstep):
    print("-------------------------------------")
    print("istep= %d | time= %.4e " %(istep,geological_time/time_scale))
    print("-------------------------------------")

    ###########################################################################
    # interpolate strain rate on particles
    ###########################################################################
    start=clock.time()

    swarm_exx=interpolate_field_on_particles(nparticle,swarm_r,swarm_s,swarm_iel,exx_nodal,icon_V)
    swarm_eyy=interpolate_field_on_particles(nparticle,swarm_r,swarm_s,swarm_iel,eyy_nodal,icon_V)
    swarm_exy=interpolate_field_on_particles(nparticle,swarm_r,swarm_s,swarm_iel,exy_nodal,icon_V)

    print("interp strain rate on particles: %.3fs" % (clock.time()-start)) ; timings[24]+=clock.time()-start

    ###########################################################################
    # interpolate temperature on particles
    ###########################################################################
    start=clock.time()

    if solve_T:
       swarm_T=interpolate_field_on_particles(nparticle,swarm_r,swarm_s,swarm_iel,T,icon_V)
       print("     -> swarm_T (m,M) %.3e %.3e " %(np.min(swarm_T),np.max(swarm_T)))
    else:
       swarm_T=0

    print("interp temperature on particles: %.3fs" % (clock.time()-start)) ; timings[25]+=clock.time()-start

    ###########################################################################
    # evaluate density and viscosity on particles (and hcond, hcapa, hprod)
    # if solve_T is false then swarm_{hcond,hcapa,hprod} are scalars equal to zero
    ###########################################################################
    start=clock.time()

    swarm_rho,swarm_eta,swarm_hcond,swarm_hcapa,swarm_hprod=\
    material_model(nparticle,swarm_mat,swarm_x,swarm_y,swarm_exx,swarm_eyy,swarm_exy,swarm_T) 

    print("     -> swarm_rho (m,M) %.5e %.5e " %(np.min(swarm_rho),np.max(swarm_rho)))
    print("     -> swarm_eta (m,M) %.5e %.5e " %(np.min(swarm_eta),np.max(swarm_eta)))

    if solve_T:
       print("     -> swarm_hcapa (m,M) %.5e %.5e " %(np.min(swarm_hcapa),np.max(swarm_hcapa)))
       print("     -> swarm_hcond (m,M) %.5e %.5e " %(np.min(swarm_hcond),np.max(swarm_hcond)))
       print("     -> swarm_hprod (m,M) %.5e %.5e " %(np.min(swarm_hprod),np.max(swarm_hprod)))

    if debug_ascii: np.savetxt('swarm_rho.ascii',np.array([swarm_x,swarm_y,swarm_rho]).T,header='# x,y,rho')
    if debug_ascii: np.savetxt('swarm_eta.ascii',np.array([swarm_x,swarm_y,swarm_eta]).T,header='# x,y,eta')

    print("compute rho,eta on particles: %.3fs" % (clock.time()-start)) ; timings[15]+=clock.time()-start

    ###########################################################################
    # project particle properties on elements 
    ###########################################################################
    start=clock.time()

    rho_elemental,eta_elemental,nparticle_elemental=\
    project_particles_on_elements(nel,nparticle,swarm_rho,swarm_eta,swarm_iel,averaging)

    if np.min(nparticle_elemental)==0: 
       exit('ABORT: an element contains no particle!')

    ptcl_stats_file.write("%d %d %d\n" % (istep,np.min(nparticle_elemental),\
                                                np.max(nparticle_elemental)))
    ptcl_stats_file.flush()

    print("     -> rho_elemental (m,M) %.3e %.3e " %(np.min(rho_elemental),np.max(rho_elemental)))
    print("     -> eta_elemental (m,M) %.3e %.3e " %(np.min(eta_elemental),np.max(eta_elemental)))

    if debug_ascii: np.savetxt('rho_elemental.ascii',np.array([xc,yc,rho_elemental]).T,header='# x,y,rho')
    if debug_ascii: np.savetxt('eta_elemental.ascii',np.array([xc,yc,eta_elemental]).T,header='# x,y,eta')

    if debug_nan and np.isnan(np.sum(rho_elemental)): exit('nan found in rho_elemental')
    if debug_nan and np.isnan(np.sum(eta_elemental)): exit('nan found in eta_elemental')

    print("project particle fields on elements: %.3fs" % (clock.time()-start)) ; timings[17]+=clock.time()-start

    ###########################################################################
    # project particle properties on nodes
    # nodal rho & eta are computed on nodes 0,1,2,3, while values on nodes
    # 4,5,6,7,8 are obtained by simple averages. In the end we obtaine Q1 fields 
    ###########################################################################
    start=clock.time()

    rho_nodal=project_particle_field_on_nodes(nel,nn_V,nparticle,swarm_rho,icon_V,swarm_iel,'arithmetic')
    eta_nodal=project_particle_field_on_nodes(nel,nn_V,nparticle,swarm_eta,icon_V,swarm_iel,averaging)

    if solve_T:
       hcond_nodal=project_particle_field_on_nodes(nel,nn_V,nparticle,swarm_hcond,icon_V,swarm_iel,'arithmetic')
       hcapa_nodal=project_particle_field_on_nodes(nel,nn_V,nparticle,swarm_hcapa,icon_V,swarm_iel,'arithmetic')

    print("     -> rho_nodal (m,M) %.3e %.3e " %(np.min(rho_nodal),np.max(rho_nodal)))
    print("     -> eta_nodal (m,M) %.3e %.3e " %(np.min(eta_nodal),np.max(eta_nodal)))

    if debug_ascii: np.savetxt('rho_nodal.ascii',np.array([x_V,y_V,rho_nodal]).T,header='# x,y,rho')
    if debug_ascii: np.savetxt('eta_nodal.ascii',np.array([x_V,y_V,eta_nodal]).T,header='# x,y,eta')

    print("project particle fields on nodes: %.3fs" % (clock.time()-start)) ; timings[18]+=clock.time()-start

    ###########################################################################
    # project nodal values onto quadrature points
    # rhoq, etaq, exxq, eyyq, exyq have size (nel,nqel)
    ###########################################################################
    start=clock.time()

    rhoq=Q1_project_nodal_field_onto_qpoints(rho_nodal,nqel,nel,N_P,icon_V)
    etaq=Q1_project_nodal_field_onto_qpoints(eta_nodal,nqel,nel,N_P,icon_V)
    exxq=Q2_project_nodal_field_onto_qpoints(exx_nodal,nqel,nel,N_V,icon_V)
    eyyq=Q2_project_nodal_field_onto_qpoints(eyy_nodal,nqel,nel,N_V,icon_V)
    exyq=Q2_project_nodal_field_onto_qpoints(exy_nodal,nqel,nel,N_V,icon_V)
    dpdxq=Q2_project_nodal_field_onto_qpoints(dpdx_nodal,nqel,nel,N_V,icon_V)
    dpdyq=Q2_project_nodal_field_onto_qpoints(dpdy_nodal,nqel,nel,N_V,icon_V)

    if solve_T:
       Tq=Q2_project_nodal_field_onto_qpoints(T,nqel,nel,N_V,icon_V)
       hcapaq=Q1_project_nodal_field_onto_qpoints(hcapa_nodal,nqel,nel,N_P,icon_V)
       hcondq=Q1_project_nodal_field_onto_qpoints(hcond_nodal,nqel,nel,N_P,icon_V)
    else:
       Tq=np.zeros((nel,nqel),dtype=np.float64)
       hcapaq=np.zeros((nel,nqel),dtype=np.float64)
       hcondq=np.zeros((nel,nqel),dtype=np.float64)

    print("     -> rhoq (m,M) %.5e %.5e " %(np.min(rhoq),np.max(rhoq)))
    print("     -> etaq (m,M) %.5e %.5e " %(np.min(etaq),np.max(etaq)))

    if solve_T:
       print("     -> Tq (m,M) %.5e %.5e " %(np.min(Tq),np.max(Tq)))
       print("     -> hcapaq (m,M) %.5e %.5e " %(np.min(hcapaq),np.max(hcapaq)))
       print("     -> hcondq (m,M) %.5e %.5e " %(np.min(hcondq),np.max(hcondq)))

    if debug_ascii: np.savetxt('rhoq.ascii',np.array([xq.flatten(),yq.flatten(),rhoq.flatten()]).T)
    if debug_ascii: np.savetxt('etaq.ascii',np.array([xq.flatten(),yq.flatten(),etaq.flatten()]).T)
    if debug_ascii and solve_T: np.savetxt('Tq.ascii',np.array([xq.flatten(),yq.flatten(),Tq.flatten()]).T)
    if debug_ascii and solve_T: np.savetxt('hcapaq.ascii',np.array([xq.flatten(),yq.flatten(),hcapaq.flatten()]).T)
    if debug_ascii and solve_T: np.savetxt('hcondq.ascii',np.array([xq.flatten(),yq.flatten(),hcondq.flatten()]).T)

    print("project nodal fields onto qpts: %.3fs" % (clock.time()-start)) ; timings[21]+=clock.time()-start

    ###########################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    ###########################################################################
    start=clock.time()

    VV_V,rhs=build_matrix_stokes(bignb_V,nel,nqel,m_V,m_P,ndof_V,Nfem_V,Nfem,\
                                 ndof_V_el,icon_V,icon_P,rhoq,etaq,JxWq,\
                                 local_to_globalV,gy,Ly,N_V,N_P,dNdx_V,dNdy_V,\
                                 eta_ref,L_ref,bc_fix_V,bc_val_V)

    if debug_nan and np.isnan(np.sum(VV_V)): exit('nan found in VV_V')

    print("build FE matrix: %.3fs" % (clock.time()-start)) ; timings[1]+=clock.time()-start

    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    sparse_matrix=sparse.coo_matrix((VV_V,(II_V,JJ_V)),shape=(Nfem,Nfem)).tocsr()

    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    print("solve time: %.3f s" % (clock.time()-start)) ; timings[2]+=clock.time()-start

    ###########################################################################
    # put solution into separate x,y velocity arrays
    ###########################################################################
    start=clock.time()

    u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
    p=sol[Nfem_V:Nfem]*(eta_ref/L_ref)

    if debug_nan and np.isnan(np.sum(u)): exit('nan found in u')
    if debug_nan and np.isnan(np.sum(v)): exit('nan found in v')
    if debug_nan and np.isnan(np.sum(p)): exit('nan found in p')

    print("     -> u (m,M) %.3e %.3e %s" %(np.min(u)/vel_scale,np.max(u)/vel_scale,vel_unit))
    print("     -> v (m,M) %.3e %.3e %s" %(np.min(v)/vel_scale,np.max(v)/vel_scale,vel_unit))
    print("     -> p (m,M) %.3e %.3e %s" %(np.min(p)/p_scale,np.max(p)/p_scale,p_unit))

    vstats_file.write("%.3e %.3e %.3e %.3e %.3e\n" % (istep,np.min(u)/vel_scale,np.max(u)/vel_scale,\
                                                            np.min(v)/vel_scale,np.max(v)/vel_scale))
    vstats_file.flush()

    if debug_ascii: np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
    if debug_ascii: np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

    print("split vel into u,v: %.3f s" % (clock.time()-start)) ; timings[14]+=clock.time()-start

    ###########################################################################
    # compute timestep
    # note that the timestep is not allowed to increase by more than 25% in one go
    ###########################################################################
    start=clock.time()

    dt1=CFLnb*(Lx/nelx)/np.max(np.sqrt(u**2+v**2))
    print('     -> dt1= %.3e %s' %(dt1/time_scale,time_unit))
    
    if solve_T:
       avrg_hcond=np.average(swarm_hcond)
       avrg_hcapa=np.average(swarm_hcapa)
       avrg_rho=np.average(swarm_rho)
       dt2=CFLnb*(Lx/nelx)**2/(avrg_hcond/avrg_hcapa/avrg_rho)
       print('     -> dt2= %.3e %s' %(dt2/time_scale,time_unit))
    else:
       dt2=1e50

    dt1=min(dt1,1.25*dt1_mem) # limiter
    dt2=min(dt2,1.25*dt2_mem) # limiter

    dt=np.min([dt1,dt2])

    geological_time+=dt

    print('     -> dt = %.3e %s' %(dt/time_scale,time_unit))
    print('     -> geological time = %e %s' %(geological_time/time_scale,time_unit))

    dt_file.write("%e %e %e %e\n" % (geological_time/time_scale,dt1/time_scale,dt2/time_scale,dt/time_scale)) 
    dt_file.flush()

    dt1_mem=dt1
    dt2_mem=dt2

    print("compute time step: %.3f s" % (clock.time()-start)) ; timings[19]+=clock.time()-start

    ###########################################################################
    # normalise pressure: simple approach to have <p> = 0 (volume or surface)
    ###########################################################################
    start=clock.time()

    match(pressure_normalisation): 
         case('surface'):
             pressure_avrg=np.sum(p[nn_P-1-(nelx+1):nn_P-1])/(nelx+1)
             p-=pressure_avrg
         case('volume'):
             pressure_avrg=0
             for iel in range(0,nel):
                 for iq in range(0,nqel):
                     pressure_avrg+=np.dot(N_P[iq,:],p[icon_P[:,iel]])*jcob*weightq[iq]
             p-=pressure_avrg/Lx/Ly
         case _ :
             exit('pressure_normalisation: unknown value')

    print("     -> p (m,M) %.3e %.3e %s" %(np.min(p),np.max(p),p_unit))

    pstats_file.write("%d %.3e %.3e\n" % (istep,np.min(p),np.max(p)))
    pstats_file.flush()

    if debug_ascii: np.savetxt('p.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

    print("normalise pressure: %.3f s" % (clock.time()-start)) ; timings[12]+=clock.time()-start

    ###########################################################################
    # project Q1 pressure onto Q2 (vel,T) mesh
    ###########################################################################
    start=clock.time()

    q=compute_nodal_pressure(m_V,nn_V,icon_V,icon_P,p,N_P_n)
    
    print("     -> q (m,M) %.3e %.3e %s" %(np.min(q),np.max(q),p_unit))

    if debug_ascii: np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')

    print("compute nodal press: %.3f s" % (clock.time()-start)) ; timings[3]+=clock.time()-start

    ###########################################################################
    # project velocity on quadrature points
    ###########################################################################
    start=clock.time()

    uq=Q2_project_nodal_field_onto_qpoints(u,nqel,nel,N_V,icon_V)
    vq=Q2_project_nodal_field_onto_qpoints(v,nqel,nel,N_V,icon_V)

    print("project vel on quad points: %.3f s" % (clock.time()-start)) ; timings[21]+=clock.time()-start

    ###########################################################################
    # build temperature matrix
    ###########################################################################
    start=clock.time()

    if solve_T: 
       VV_T,rhs=build_matrix_energy(bignb_T,nel,nqel,m_T,Nfem_T,T,icon_V,rhoq,etaq,Tq,uq,vq,\
                                    hcondq,hcapaq,exxq,eyyq,exyq,dpdxq,dpdyq,JxWq,N_V,dNdx_V,dNdy_V,\
                                    bc_fix_T,bc_val_T,dt,formulation,rho0)

       print("build FE matrix : %.3f s" % (clock.time()-start)) ; timings[4]+=clock.time()-start

       ###########################################################################
       # solve system
       ###########################################################################
       start=clock.time()

       sparse_matrix=sparse.coo_matrix((VV_T,(II_T,JJ_T)),shape=(Nfem_T,Nfem_T)).tocsr()

       T=sps.linalg.spsolve(sparse_matrix,rhs)

       if debug_nan and np.isnan(np.sum(T)): exit('nan found in T')

       print("     -> T (m,M) %.3e %.3e " %(np.min(T),np.max(T)))

       if debug_ascii: np.savetxt('T.ascii',np.array([x_V,y_V,T]).T,header='# x,y,T')

       Tstats_file.write("%.3e %.3e %.3e\n" %(istep,np.min(T)-TKelvin,np.max(T)-TKelvin)) 
       Tstats_file.flush()

       print("solve T time: %.3f s" % (clock.time()-start)) ; timings[5]+=clock.time()-start

    #end if solve_T

    ###########################################################################
    # compute vrms 
    ###########################################################################
    start=clock.time()

    vrms,EK,WAG,TVD,GPE,ITE,TM=\
    global_quantities(nel,nqel,xq,yq,uq,vq,Tq,rhoq,hcapaq,etaq,exxq,eyyq,exyq,Lx,Ly,JxWq,gy)

    vrms_file.write("%e %e \n" % (geological_time/time_scale,vrms/vel_scale)) ; vrms_file.flush()
    TM_file.write("%e %e \n" % (geological_time/time_scale,TM)) ; TM_file.flush()
    EK_file.write("%e %e \n" % (geological_time/time_scale,EK)) ; EK_file.flush()
    TVD_file.write("%e %e \n" % (geological_time/time_scale,TVD)) ; TVD_file.flush()

    print("     istep= %.6d ; vrms   = %.3e %s" %(istep,vrms/vel_scale,vel_unit))

    print("compute global quantities: %.3f s" % (clock.time()-start)) ; timings[6]+=clock.time()-start

    ###########################################################################
    # compute nodal heat flux 
    ###########################################################################
    start=clock.time()

    if solve_T: 
       dTdx_nodal,dTdy_nodal,qx_nodal,qy_nodal=\
       compute_nodal_heat_flux(icon_V,T,hcond_nodal,nn_V,m_V,nel,dNdx_V_n,dNdy_V_n)

       print("     -> dTdx_nodal (m,M) %.3e %.3e " %(np.min(dTdx_nodal),np.max(dTdx_nodal)))
       print("     -> dTdy_nodal (m,M) %.3e %.3e " %(np.min(dTdy_nodal),np.max(dTdy_nodal)))
       print("     -> qx_nodal (m,M) %.3e %.3e " %(np.min(qx_nodal),np.max(qx_nodal)))
       print("     -> qy_nodal (m,M) %.3e %.3e " %(np.min(qy_nodal),np.max(qy_nodal)))

       qx0=qx_nodal[cornerBL] ; qy0=qy_nodal[cornerBL]
       qx1=qx_nodal[cornerBR] ; qy1=qy_nodal[cornerBR]
       qx2=qx_nodal[cornerTR] ; qy2=qy_nodal[cornerTR]
       qx3=qx_nodal[cornerTL] ; qy3=qy_nodal[cornerTL]

       corner_q_file.write("%e %e %e %e %e %e %e %e %e\n" % (geological_time/time_scale,qx1,qy1,qx2,qy2,qx3,qy3,qx4,qy4)) 
       corner_q_file.flush()

    else:
       qx_nodal=0 
       qy_nodal=0 

    print("compute nodal heat flux: %.3f s" % (clock.time()-start)) ; timings[7]+=clock.time()-start

    ###########################################################################
    # compute heat flux and Nusselt at top and bottom
    ###########################################################################
    start=clock.time()

    if istep%every_Nu==0 and solve_T: 

       avrg_T_bot,avrg_T_top,avrg_dTdy_bot,avrg_dTdy_top,Nu=\
       compute_Nu(Lx,Ly,nel,top_element,bot_element,icon_V,T,dTdy_nodal,nqperdim,qcoords,qweights,hx)

       print("     -> <T> (bot,top)= %.3e %.3e " %(avrg_T_bot,avrg_T_top))
       print("     -> <dTdy> (bot,top)= %.3e %.3e " %(avrg_dTdy_bot,avrg_dTdy_top))
       print("     -> Nusselt= %.3e " %(Nu))

       Nu_file.write("%e %e \n" % (geological_time/time_scale,Nu)) ; Nu_file.flush()
       avrg_T_bot_file.write("%e %e \n" % (geological_time/time_scale,avrg_T_bot)) ; avrg_T_bot_file.flush()
       avrg_T_top_file.write("%e %e \n" % (geological_time/time_scale,avrg_T_top)) ; avrg_T_top_file.flush()
       avrg_dTdy_bot_file.write("%e %e \n" % (geological_time/time_scale,avrg_dTdy_bot)) ; avrg_dTdy_bot_file.flush()
       avrg_dTdy_top_file.write("%e %e \n" % (geological_time/time_scale,avrg_dTdy_top)) ; avrg_dTdy_top_file.flush()

       print("compute Nu: %.3f s" % (clock.time()-start)) ; timings[8]+=clock.time()-start

    ###########################################################################
    # compute temperature profile
    ###########################################################################
    start=clock.time()

    if istep%2500==0 and solve_T: 

       T_profile=np.zeros(nny,dtype=np.float64)  
       y_profile=np.zeros(nny,dtype=np.float64)  

       counter=0    
       for j in range(0,nny):
           for i in range(0,nnx):
               T_profile[j]+=T[counter]/nnx
               y_profile[j]=y_V[counter]
               counter+=1
           #end for
       #end for

       np.savetxt('T_profile_'+str(istep)+'.ascii',np.array([y_profile,T_profile]).T,header='#y,T')

       print("compute T profile: %.3f s" % (clock.time()-start)) ; timings[9]+=clock.time()-start

    ###########################################################################
    # compute nodal strainrate
    ###########################################################################
    start=clock.time()

    exx_nodal,eyy_nodal,exy_nodal,e_nodal=compute_nodal_strain_rate(icon_V,u,v,nn_V,m_V,nel,dNdx_V_n,dNdy_V_n)

    print("     -> exx_nodal (m,M) %.3e %.3e " %(np.min(exx_nodal),np.max(exx_nodal)))
    print("     -> eyy_nodal (m,M) %.3e %.3e " %(np.min(eyy_nodal),np.max(eyy_nodal)))
    print("     -> exy_nodal (m,M) %.3e %.3e " %(np.min(exy_nodal),np.max(exy_nodal)))

    if debug_ascii: np.savetxt('strainrate.ascii',np.array([x_V,y_V,exx_nodal,eyy_nodal,exy_nodal]).T)

    print("compute nodal sr: %.3f s" % (clock.time()-start)) ; timings[11]+=clock.time()-start

    ###########################################################################
    # compute full stress 
    ###########################################################################
    ## dev strain rate ?!

    tauxx_nodal=2*eta_nodal*exx_nodal
    tauyy_nodal=2*eta_nodal*eyy_nodal
    tauxy_nodal=2*eta_nodal*exy_nodal

    sigmaxx_nodal=-q+tauxx_nodal
    sigmayy_nodal=-q+tauyy_nodal
    sigmaxy_nodal=   tauxy_nodal

    np.savetxt('top_eyy_'+str(istep)+'.ascii',np.array([x_V[top_nodes],eyy_nodal[top_nodes]]).T)
    np.savetxt('bot_eyy_'+str(istep)+'.ascii',np.array([x_V[bot_nodes],eyy_nodal[bot_nodes]]).T)
    np.savetxt('top_tauyy_'+str(istep)+'.ascii',np.array([x_V[top_nodes],tauyy_nodal[top_nodes]]).T)
    np.savetxt('bot_tauyy_'+str(istep)+'.ascii',np.array([x_V[bot_nodes],tauyy_nodal[bot_nodes]]).T)
    np.savetxt('top_sigmayy_'+str(istep)+'.ascii',np.array([x_V[top_nodes],sigmayy_nodal[top_nodes]]).T)
    np.savetxt('bot_sigmayy_'+str(istep)+'.ascii',np.array([x_V[bot_nodes],sigmayy_nodal[bot_nodes]]).T)

    ###########################################################################
    # compute dynamic topography at bottom and surface topo 
    ###########################################################################

    dyn_topo_top=np.zeros(2*nelx+1,dtype=np.float64)
    dyn_topo_bot=np.zeros(2*nelx+1,dtype=np.float64)

    if np.all(rho_nodal[top_nodes])>0 and abs(gy)>0:
       avrg_sigmayy=np.average(sigmayy_nodal[top_nodes])
       dyn_topo_top=(sigmayy_nodal[top_nodes]-avrg_sigmayy)/gy/(rho_nodal[top_nodes]-rho_air)
       np.savetxt('dynamic_topography_top_'+str(istep)+'.ascii',np.array([x_V[top_nodes],dyn_topo_top]).T)

    if np.all(rho_nodal[bot_nodes])>0 and abs(gy)>0:
       avrg_sigmayy=np.average(sigmayy_nodal[bot_nodes])
       dyn_topo_bot=(sigmayy_nodal[bot_nodes]-avrg_sigmayy)/gy/(rho_nodal[bot_nodes]-rho_core)
       np.savetxt('dynamic_topography_bot_'+str(istep)+'.ascii',np.array([x_V[bot_nodes],dyn_topo_bot]).T)

    ###########################################################################
    # compute nodal pressure gradient 
    ###########################################################################
    start=clock.time()

    dpdx_nodal,dpdy_nodal=compute_nodal_pressure_gradient(icon_V,q,nn_V,m_V,nel,dNdx_V_n,dNdy_V_n)

    print("     -> dpdx_nodal (m,M) %.3e %.3e " %(np.min(dpdx_nodal),np.max(dpdx_nodal)))
    print("     -> dpdy_nodal (m,M) %.3e %.3e " %(np.min(dpdy_nodal),np.max(dpdy_nodal)))

    if debug_ascii: np.savetxt('pressure_gradient.ascii',np.array([x_V,y_V,dpdx_nodal,dpdy_nodal]).T)

    print("compute nodal pressure gradient: %.3f s" % (clock.time()-start)) ; timings[8]+=clock.time()-start

    ###########################################################################
    # advect particles
    ###########################################################################
    start=clock.time()

    swarm_x,swarm_y,swarm_u,swarm_v,swarm_active=\
    advect_particles(RKorder,dt,nparticle,swarm_x,swarm_y,swarm_active,\
                     u,v,Lx,Ly,hx,hy,nelx,nely,icon_V,x_V,y_V)

    print("     -> nb inactive particles:",nparticle-np.sum(swarm_active))
    print("     -> swarm_x (m,M) %.3e %.3e " %(np.min(swarm_x),np.max(swarm_x)))
    print("     -> swarm_y (m,M) %.3e %.3e " %(np.min(swarm_y),np.max(swarm_y)))


    print("advect particles: %.3f s" % (clock.time()-start)) ; timings[13]+=clock.time()-start

    ###########################################################################
    # locate particles and compute reduced coordinates
    ###########################################################################
    start=clock.time()

    swarm_r,swarm_s,swarm_iel=locate_particles(nparticle,swarm_x,swarm_y,hx,hy,x_V,y_V,icon_V,nelx)

    print("locate particles: %.3fs" % (clock.time()-start)) ; timings[16]+=clock.time()-start

    ###########################################################################
    # export min/max coordinates of each material in one single file
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
        ymin=np.min(swarm_y[swarm_mat==i]) ; mats[counter]=ymin ; counter+=1
        ymax=np.max(swarm_y[swarm_mat==i]) ; mats[counter]=ymax ; counter+=1

    mats.tofile(mats_file,sep=' ',format='%.4e ') ; mats_file.write('\n')
    mats_file.flush()

    print("write min/max extents: %.3fs" % (clock.time()-start)) #; timings[16]+=clock.time()-start

    ###########################################################################
    # generate/write in pvd files
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
    # plot of solution
    ###########################################################################
    start=clock.time()

    if istep%every_solution_vtu==0 or istep==nstep-1: 
       export_solution_to_vtu(istep,nel,nn_V,m_V,solve_T,vel_scale,TKelvin,x_V,y_V,\
                              u,v,q,T,eta_nodal,rho_nodal,exx_nodal,eyy_nodal,\
                              exy_nodal,e_nodal,qx_nodal,qy_nodal,rho_elemental,\
                              sigmaxx_nodal,sigmayy_nodal,sigmaxy_nodal,\
                              eta_elemental,nparticle_elemental,icon_V)

       print("export solution to vtu file: %.3f s" % (clock.time()-start)) ; timings[10]+=clock.time()-start

    ########################################################################
    # export particles to vtu file
    ########################################################################
    start=clock.time()

    if istep%every_swarm_vtu==0 or istep==nstep-1: 
       export_swarm_to_vtu(istep,nparticle,solve_T,vel_scale,swarm_x,swarm_y,\
                           swarm_u,swarm_v,swarm_mat,swarm_rho,swarm_eta,\
                           swarm_paint,swarm_exx,swarm_eyy,swarm_exy,swarm_T,\
                           swarm_hcond,swarm_hcapa) 

       print("export particles to vtu file: %.3f s" % (clock.time()-start)) ; timings[20]+=clock.time()-start

    ########################################################################
    # export quadrature points to vtu file
    ########################################################################
    start=clock.time()

    if istep%every_quadpoints_vtu==0 or istep==nstep-1: 
       export_quadpoints_to_vtu(istep,nel,nqel,nq,solve_T,xq,yq,rhoq,etaq,Tq,hcondq,hcapaq,dpdxq,dpdyq)

       print("export quad pts to vtu file: %.3f s" % (clock.time()-start)) ; timings[22]+=clock.time()-start

    ########################################################################
    # compute gravitational field above domain 
    # xs[npts],ys: coordinates of satellite
    # gxI,gyI,gnormI: gravity from internal density distribution
    # gxDTt,gyDTt,gnormDTt: gravity from dynamic topography at top 
    # gxDTb,gyDTb,gnormDTb: gravity from dynamic topography at bottom
    ########################################################################
    start=clock.time()
  
    npts=256   
    rho_ref=0
    ys=Ly+200e3

    if istep==0:
       xs=np.zeros(npts,dtype=np.float64)  
       gxI=np.zeros((npts,nstep),dtype=np.float64)  
       gyI=np.zeros((npts,nstep),dtype=np.float64)  
       gnormI=np.zeros((npts,nstep),dtype=np.float64)  
       gnormI_rate=np.zeros((npts,nstep),dtype=np.float64)  
       gxDTt=np.zeros((npts,nstep),dtype=np.float64)  
       gyDTt=np.zeros((npts,nstep),dtype=np.float64)  
       gnormDTt=np.zeros((npts,nstep),dtype=np.float64)  
       gnormDTt_rate=np.zeros((npts,nstep),dtype=np.float64)  
       gxDTb=np.zeros((npts,nstep),dtype=np.float64)  
       gyDTb=np.zeros((npts,nstep),dtype=np.float64)  
       gnormDTb=np.zeros((npts,nstep),dtype=np.float64)  
       gnormDTb_rate=np.zeros((npts,nstep),dtype=np.float64)  

    for i in range(0,npts):
        xs[i]=i*Lx/(npts-1)
        gxI[i,istep],gyI[i,istep],gnormI[i,istep]=\
        compute_gravity_at_point(xs[i],ys,nel,xc,yc,rho_elemental,hx,hy,rho_ref)

        gxDTt[i,istep],gyDTt[i,istep],gnormDTt[i,istep]=\
        compute_gravity_fromDT_at_point(xs[i],ys,Ly,nelx,x_V[top_nodes],rho_nodal[top_nodes],dyn_topo_top,rho_air)

        gxDTb[i,istep],gyDTb[i,istep],gnormDTb[i,istep]=\
        compute_gravity_fromDT_at_point(xs[i],ys,0,nelx,x_V[bot_nodes],rho_nodal[bot_nodes],dyn_topo_bot,rho_core)

    np.savetxt('gravityI_'+str(istep)+'.ascii',np.array([xs,gnormI[:,istep],gxI[:,istep],gyI[:,istep]]).T,header='#x,g,gx,gy')
    np.savetxt('gravityDTt_'+str(istep)+'.ascii',np.array([xs,gnormDTt[:,istep],gxDTt[:,istep],gyDTt[:,istep]]).T,header='#x,g,gx,gy')
    np.savetxt('gravityDTb_'+str(istep)+'.ascii',np.array([xs,gnormDTb[:,istep],gxDTb[:,istep],gyDTb[:,istep]]).T,header='#x,g,gx,gy')

    if istep>0:
       gnormI_rate[:,istep]=(gnormI[:,istep]-gnormI[:,istep-1])/dt
       gnormDTt_rate[:,istep]=(gnormDTt[:,istep]-gnormDTt[:,istep-1])/dt
       gnormDTb_rate[:,istep]=(gnormDTb[:,istep]-gnormDTb[:,istep-1])/dt
       np.savetxt('gravityI_rate_'+str(istep)+'.ascii',np.array([xs,gnormI_rate[:,istep]/mGal*year]).T,header='#x,g')
       np.savetxt('gravityDTt_rate_'+str(istep)+'.ascii',np.array([xs,gnormDTt_rate[:,istep]/mGal*year]).T,header='#x,g')
       np.savetxt('gravityDTb_rate_'+str(istep)+'.ascii',np.array([xs,gnormDTb_rate[:,istep]/mGal*year]).T,header='#x,g')

    print("compute gravity: %.3f s" % (clock.time()-start)) ; timings[23]+=clock.time()-start

    ###########################################################################
    # assess steady state
    ###########################################################################
    start=clock.time()

    steady_state_u=np.linalg.norm(u_mem-u,2)/np.linalg.norm(u,2)<tol_ss
    steady_state_v=np.linalg.norm(v_mem-v,2)/np.linalg.norm(v,2)<tol_ss
    steady_state_p=np.linalg.norm(p_mem-p,2)/np.linalg.norm(p,2)<tol_ss

    if solve_T:
       steady_state_T=np.linalg.norm(T_mem-T,2)/np.linalg.norm(T,2)<tol_ss
       print('     -> steady state u,v,p,T',steady_state_u,steady_state_v,\
                                            steady_state_p,steady_state_T)
    else:
       steady_state_T=True
       print('     -> steady state u,v,p',steady_state_u,steady_state_v,\
                                          steady_state_p)

    steady_state=steady_state_u and steady_state_v and\
                 steady_state_p and steady_state_T 

    u_mem=u.copy()
    v_mem=v.copy()
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
       print("comp. glob quantities: %8.3f s  (%.3f s per call) | %5.2f percent" % (timings[6],timings[6]/(istep+1),timings[6]/duration*100))
       print("comp. nodal p: %8.3f s          (%.3f s per call) | %5.2f percent" % (timings[3],timings[3]/(istep+1),timings[3]/duration*100))
       print("comp. nodal sr: %8.3f s         (%.3f s per call) | %5.2f percent" % (timings[11],timings[11]/(istep+1),timings[11]/duration*100))
       print("comp. nodal heat flux: %8.3f s  (%.3f s per call) | %5.2f percent" % (timings[7],timings[7]/(istep+1),timings[7]/duration*100))
       print("comp. T profile: %8.3f s        (%.3f s per call) | %5.2f percent" % (timings[9],timings[9]/(istep+1),timings[9]/duration*100)) 
       print("comp. nodal press grad: %8.3f s (%.3f s per call) | %5.2f percent" % (timings[8],timings[8]/(istep+1),timings[8]/duration*100))
       print("normalise pressure: %8.3f s     (%.3f s per call) | %5.2f percent" % (timings[12],timings[12]/(istep+1),timings[12]/duration*100))
       print("advect particles: %8.3f s       (%.3f s per call) | %5.2f percent" % (timings[13],timings[13]/(istep+1),timings[13]/duration*100))
       print("split solution: %8.3f s         (%.3f s per call) | %5.2f percent" % (timings[14],timings[14]/(istep+1),timings[14]/duration*100))
       print("compute swarm rho,eta: %8.3f s  (%.3f s per call) | %5.2f percent" % (timings[15],timings[15]/(istep+1),timings[15]/duration*100))
       print("locate particles: %8.3f s       (%.3f s per call) | %5.2f percent" % (timings[16],timings[16]/(istep+1),timings[16]/duration*100))
       print("comp eltal rho,eta: %8.3f s     (%.3f s per call) | %5.2f percent" % (timings[17],timings[17]/(istep+1),timings[17]/duration*100))
       print("comp nodal rho,eta: %8.3f s     (%.3f s per call) | %5.2f percent" % (timings[18],timings[18]/(istep+1),timings[18]/duration*100))
       print("comp timestep: %8.3f s          (%.3f s per call) | %5.2f percent" % (timings[19],timings[19]/(istep+1),timings[19]/duration*100))
       print("export solution to vtu: %8.3f s (%.3f s per call) | %5.2f percent" % (timings[10],timings[10]/(istep+1),timings[10]/duration*100))
       print("export swarm to vtu: %8.3f s    (%.3f s per call) | %5.2f percent" % (timings[20],timings[20]/(istep+1),timings[20]/duration*100))
       print("export qpts to vtu: %8.3f s     (%.3f s per call) | %5.2f percent" % (timings[22],timings[22]/(istep+1),timings[22]/duration*100))
       print("project fields on qpts: %8.3f s (%.3f s per call) | %5.2f percent" % (timings[21],timings[21]/(istep+1),timings[21]/duration*100))
       print("compute gravity: %8.3f s        (%.3f s per call) | %5.2f percent" % (timings[23],timings[23]/(istep+1),timings[23]/duration*100))
       print("interp sr on ptcls: %8.3f s     (%.3f s per call) | %5.2f percent" % (timings[24],timings[24]/(istep+1),timings[24]/duration*100))
       print("interp T on ptcls: %8.3f s      (%.3f s per call) | %5.2f percent" % (timings[25],timings[25]/(istep+1),timings[25]/duration*100))
       print("----------------------------------------------------------------------")
       print("compute time per timestep: %.2f" %(duration/(istep+1)))
       print("----------------------------------------------------------------------")

    dtimings=timings-timings_mem
    dtimings[0]=istep ; dtimings.tofile(timings_file,sep=' ',format='%e') ; timings_file.write(" \n" ) 
    timings_mem[:]=timings[:]

    ###########################################################################

    if geological_time>end_time: 
       print('***** end time reached *****')
       break

    if steady_state: 
       print('***** steady state reached *****')
       break

#end for istep

pvd_solution_file.write('  </Collection>\n')
pvd_solution_file.write('</VTKFile>\n')
pvd_swarm_file.write('  </Collection>\n')
pvd_swarm_file.write('</VTKFile>\n')

###########################################################################
# export horizontal and vertical profiles
###########################################################################
start=clock.time()

np.savetxt('profile_vertical.ascii',np.array([y_V[middleV_nodes],\
                                              u[middleV_nodes],\
                                              v[middleV_nodes],\
                                              q[middleV_nodes],\
                                              T[middleV_nodes]]).T)

np.savetxt('profile_horizontal.ascii',np.array([x_V[middleH_nodes],\
                                                  u[middleH_nodes],\
                                                  v[middleH_nodes],\
                                                  q[middleH_nodes],\
                                                  T[middleH_nodes]]).T)

###############################################################################
# close files
###############################################################################
       
vstats_file.close()
pstats_file.close()
vrms_file.close()
dt_file.close()
Nu_file.close()
TM_file.close()
EK_file.close()
TVD_file.close()
corner_q_file.close()

###############################################################################

print("-----------------------------")
print("total compute time: %.1f s" % (duration))
print("sum timings: %.1f s" % (np.sum(timings)))
print("-----------------------------")
    
###############################################################################
