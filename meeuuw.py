import numpy as np
import sys as sys
import numba
import random
import time as clock
import scipy.sparse as sps
from scipy import sparse
from PoissonDisc import *
from compute_vrms import *
from pic_functions import *
from basis_functions import *
from compute_nodal_strain_rate import *
from compute_nodal_heat_flux import *
from compute_nodal_pressure import *
from export_swarm_to_vtu import *
from export_solution_to_vtu import *
from export_quadpoints_to_vtu import *
from project_nodal_field_onto_qpoints import *

###############################################################################
# constants

cm=0.01
eps=1e-9
year=365.25*3600*24
Myear=365.25*3600*24*1e6

print("-----------------------------")
print("----------- MEEUUW ----------")
print("-----------------------------")

###############################################################################
# experiment 0: Blankenbach et al, 1993    - isoviscous convection
# experiment 1: van Keken et al, JGR, 1997 - Rayleigh-Taylor experiment
# experiment 2: Schmeling et al, PEPI 2008 - Newtonian subduction
# experiment 3: Tosi et al, 2015           - visco-plastic convection
###############################################################################

experiment=2

match(experiment):
     case 0 | 3:
         Lx=1
         Ly=1
         eta_ref=1
         solve_T=True
         vel_scale=1 ; vel_unit=' '
         time_scale=1
         p_scale=1
         Ttop=0
         Tbottom=1
         alphaT=1e-2   # thermal expansion coefficient
         hcond=1       # thermal conductivity
         hcapa=1       # heat capacity
         rho0=1
         Ra=1e4
         gy=-Ra/alphaT 
         TKelvin=0
         pressure_normalisation='surface'
         every_Nu=1
         if experiment==3: 
            import tosi
            case_tosi=1
            gamma_T=np.log(1e5)
            eta_star=1e-3 
            eta_ref=1e-2
            alphaT=1e-4
            Ra,sigma_y,gamma_y=tosi.assign_parameters(case_tosi)
            eta_min=1e-5
            eta_max=1
           
     case 1 :
         Lx=0.9142
         Ly=1
         gy=-10 
         eta_ref=100
         solve_T=False
         vel_scale=1
         p_scale=1
         time_scale=1
         pressure_normalisation='volume'
         every_Nu=1000
         TKelvin=0
     case 2 :
         eta_ref=1e21
         solve_T=False
         p_scale=1e6 ; p_unit="MPa"
         vel_scale=cm/year ; vel_unit='cm/yr'
         time_scale=year ; time_unit='yr'
         every_Nu=100000
         TKelvin=0
         pressure_normalisation='surface'

         Lx=3000e3
         Ly=750e3
         gy=-9.81

     case _ :
         exit('setup - unknown experiment')  

if int(len(sys.argv)==4):
   nelx  = int(sys.argv[1])
   nely  = int(sys.argv[2])
   nstep = int(sys.argv[3])
else:
   nelx=200
   nely=50
   nstep=1

CFLnb=0.5
         
every_vtu=5

RKorder=4
nparticle_per_dim=10
particle_distribution=0 # 0: random, 1: reg, 2: Poisson Disc
#averaging='arithmetic'
#averaging='geometric'
averaging='harmonic'
#use_nodal_rho=True
#use_nodal_eta=True

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

EBA=False

debug_ascii=False
debug_nan=False

timings=np.zeros(22+1)

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

vrms_file=open('vrms.ascii',"w")
vrms_file.write("#time,vrms\n")
pstats_file=open('pressure_stats.ascii',"w")
pstats_file.write("#istep,min p, max p\n")
vstats_file=open('velocity_stats.ascii',"w")
vstats_file.write("#istep,min(u),max(u),min(v),max(v)\n")
dt_file=open('dt.ascii',"w")
dt_file.write("#time dt1 dt2 dt\n")
ptcl_stats_file=open('particle_stats.ascii',"w")
Nu_file=open('Nu.ascii',"w")

###############################################################################

print('experiment=',experiment)
print('Lx=',Lx)
print('Ly=',Ly)
print('nn_V=',nn_V)
print('nn_P=',nn_P)
print('nel=',nel)
print('Nfem_V=',Nfem_V)
print('Nfem_P=',Nfem_P)
print('Nfem=',Nfem)
print('nqperdim=',nqperdim)
print('particle_distribution=',particle_distribution)
print('RKorder=',RKorder)
#print('use_nodal_rho=',use_nodal_rho)
#print('use_nodal_eta=',use_nodal_eta)
print("-----------------------------")

###############################################################################
# build velocity nodes coordinates 
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64) # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64) # y coordinates

counter=0    
for j in range(0,2*nely+1):
    for i in range(0,2*nelx+1):
        x_V[counter]=i*hx/2
        y_V[counter]=j*hy/2
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

nnx=2*nelx+1 
nny=2*nely+1 

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=(i)*2+1+(j)*2*nnx -1
        icon_V[1,counter]=(i)*2+3+(j)*2*nnx -1
        icon_V[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        icon_V[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        icon_V[4,counter]=(i)*2+2+(j)*2*nnx -1
        icon_V[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        icon_V[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        icon_V[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        icon_V[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter+=1
    #end for
#end for

print("build icon_V: %.3f s" % (clock.time()-start))

###############################################################################
# build pressure grid 
###############################################################################
start=clock.time()

x_P=np.zeros(nn_P,dtype=np.float64) # x coordinates
y_P=np.zeros(nn_P,dtype=np.float64) # y coordinates

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
start = clock.time()

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
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool) # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

match(experiment):

     case 0 | 2 | 3 : # Blankenbach et al convection, Tosi et al 2015, free slip all sides
         for i in range(0,nn_V):
             if x_V[i]/Lx<eps:
                bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
             if x_V[i]/Lx>(1-eps):
                bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
             if y_V[i]/Ly<eps:
                bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
             if y_V[i]/Ly>(1-eps):
                bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

     case 1 : # van Keken et al Rayleigh-Taylor instability, no slip top, bottom 
         for i in range(0,nn_V):
             if x_V[i]/Lx<eps:
                bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
             if x_V[i]/Lx>(1-eps):
                bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
             if y_V[i]/Ly<eps:
                bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
                bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
             if y_V[i]/Ly>(1-eps):
                bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
                bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
     case _ :
         exit('bc_V - unknown experiment')  

print("velocity b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# define temperature boundary conditions
###############################################################################
start=clock.time()

if solve_T:

   bc_fix_T=np.zeros(Nfem_T,dtype=bool)  
   bc_val_T=np.zeros(Nfem_T,dtype=np.float64) 

   match(experiment):
        case 0 | 3 :
            for i in range(0,nn_V):
                if y_V[i]<eps:
                   bc_fix_T[i]=True ; bc_val_T[i]=Tbottom
                if y_V[i]>(Ly-eps):
                   bc_fix_T[i]=True ; bc_val_T[i]=Ttop
        case _:
            exit('bc_T - unknown experiment')  

   print("temperature b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# initial temperature
###############################################################################
start=clock.time()

T=np.zeros(nn_V,dtype=np.float64)

if solve_T:

   match(experiment):
        case 0 | 3 :
            for i in range(0,nn_V):
                T[i]=(Tbottom-Ttop)*(Ly-y_V[i])/Ly+Ttop\
                    +0.01*np.cos(np.pi*x_V[i]/Lx)*np.sin(np.pi*y_V[i]/Ly)
        case _:
            exit('unknown experiment')  

   T_mem=T.copy()

   if debug_ascii: np.savetxt('temperature_init.ascii',np.array([x_V,y_V,T]).T,header='# x,y,T')

   print("     -> T init (m,M) %.3e %.3e " %(np.min(T),np.max(T)))

   print("initial temperature: %.3f s" % (clock.time()-start))

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
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (clock.time() - start))

###############################################################################
# Compute jacobian matrix (inverse and determinant)
# This is only valid for rectangular elements.
###############################################################################

jcbi=np.zeros((ndim,ndim),dtype=np.float64)
jcbi[0,0]=2/hx
jcbi[1,1]=2/hy
jcob=hx*hy/4

###############################################################################
# precompute basis functions values at q points
###############################################################################
start=clock.time()

rq=np.zeros(nqel,dtype=np.float64) 
sq=np.zeros(nqel,dtype=np.float64) 
weightq=np.zeros(nqel,dtype=np.float64) 
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
        N_V[counterq,0:m_V]=basis_functions_V(rq[counterq],sq[counterq])
        N_P[counterq,0:m_P]=basis_functions_P(rq[counterq],sq[counterq])
        dNdr_V[counterq,0:m_V]=basis_functions_V_dr(rq[counterq],sq[counterq])
        dNds_V[counterq,0:m_V]=basis_functions_V_ds(rq[counterq],sq[counterq])
        dNdx_V[counterq,0:m_V]=jcbi[0,0]*dNdr_V[counterq,0:m_V]
        dNdy_V[counterq,0:m_V]=jcbi[1,1]*dNds_V[counterq,0:m_V]
        counterq+=1

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
# fill I,J arrays
###############################################################################
start=clock.time()

bignb=nel*( (m_V*ndof_V)**2 + 2*(m_V*ndof_V*m_P) )

II_V=np.zeros(bignb,dtype=np.int32)    
JJ_V=np.zeros(bignb,dtype=np.int32)    
VV_V=np.zeros(bignb,dtype=np.float64)    

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
# fill I,J arrays
###############################################################################
start=clock.time()

if solve_T:

   bignb=nel*m_T**2 

   II_T=np.zeros(bignb,dtype=np.int32)    
   JJ_T=np.zeros(bignb,dtype=np.int32)    
   VV_T=np.zeros(bignb,dtype=np.float64)    

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

nparticle_per_element=nparticle_per_dim**2
nparticle=nel*nparticle_per_element

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

         swarm_r,swarm_s,swarm_iel=locate_particles(nparticle,swarm_x,swarm_y,hx,hy,x_V,y_V,icon_V,nelx)

     case _ :
         exit('unknown particle_distribution')

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

swarm_mat=np.zeros(nparticle,dtype=np.int32)

match(experiment):
     case 0 | 3 :
         swarm_mat[:]=1
     case 1 :
         for im in range (0,nparticle):
             if swarm_y[im]<0.2+0.02*np.cos(swarm_x[im]*np.pi/0.9142):
                swarm_mat[im]=1
             else:
                swarm_mat[im]=2
     case 2 :
         swarm_mat[:]=2 # mantle 
         for ip in range(0,nparticle):
             if swarm_y[ip]>Ly-50e3:
                swarm_mat[ip]=1 # sticky air
             if swarm_x[ip]>1000e3 and swarm_y[ip]<Ly-50e3 and swarm_y[ip]>Ly-150e3: 
                swarm_mat[ip]=3 # lithosphere
             if swarm_x[ip]>1000e3 and swarm_x[ip]<1100e3 and\
                swarm_y[ip]>Ly-250e3 and swarm_y[ip]<Ly-50e3:
                swarm_mat[ip]=3 # lithosphere

     case _ :
         exit('mat - unknown experiment')  

print("     -> swarm_mat (m,M) %.3e %.3e " %(np.min(swarm_mat),np.max(swarm_mat)))

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
       
exx_nodal=np.zeros(nn_V,dtype=np.float64)  
eyy_nodal=np.zeros(nn_V,dtype=np.float64)  
exy_nodal=np.zeros(nn_V,dtype=np.float64)  

topstart=clock.time()

for istep in range(0,nstep):
    print("-------------------------------------")
    print("istep= %d | time= %.4e " %(istep,geological_time))
    print("-------------------------------------")

    ###########################################################################
    # evaluate density and viscosity on particles (and T, hcond, hcapa)
    ###########################################################################
    start=clock.time()

    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)

    if solve_T:
       swarm_hcond=np.zeros(nparticle,dtype=np.float64)
       swarm_hcapa=np.zeros(nparticle,dtype=np.float64)
       swarm_T=interpolate_field_on_particles(nparticle,swarm_r,swarm_s,swarm_iel,T,icon_V)
       print("     -> swarm_T (m,M) %.3e %.3e " %(np.min(swarm_T),np.max(swarm_T)))
    else:
       swarm_T=0
       swarm_hcond=0
       swarm_hcapa=0

    swarm_exx=interpolate_field_on_particles(nparticle,swarm_r,swarm_s,swarm_iel,exx_nodal,icon_V)
    swarm_eyy=interpolate_field_on_particles(nparticle,swarm_r,swarm_s,swarm_iel,eyy_nodal,icon_V)
    swarm_exy=interpolate_field_on_particles(nparticle,swarm_r,swarm_s,swarm_iel,exy_nodal,icon_V)

    match(experiment):
         case 0 :
             swarm_rho[:]=rho0*(1-alphaT*swarm_T[:])
             swarm_eta[:]=1
             swarm_hcond[:]=1
             swarm_hcapa[:]=1
         case 1 :
             swarm_eta[:]=100
             for ip in range(0,nparticle):
                 if swarm_mat[ip]==1:
                    swarm_rho[ip]=1000
                 else:
                    swarm_rho[ip]=1010
         case 2:
             for ip in range(0,nparticle):
                 match(swarm_mat[ip]):
                      case 1 :
                            swarm_rho[ip]=0
                            swarm_eta[ip]=1e19
                      case 2 :
                            swarm_rho[ip]=3200
                            swarm_eta[ip]=1e21
                      case 3 :
                            swarm_rho[ip]=3300
                            swarm_eta[ip]=1e23
                      case _ :
                            exit('Abort: swarm_mat unknown')   
             
         case 3 :
             swarm_rho[:]=rho0*(1-alphaT*swarm_T[:])
             for ip in range(0,nparticle):
                 swarm_eta[ip]=tosi.viscosity(swarm_T[ip],swarm_exx[ip],swarm_eyy[ip],swarm_exy[ip],swarm_y[ip],\
                                              gamma_T,gamma_y,sigma_y,eta_star,case_tosi)
             swarm_hcond[:]=1
             swarm_hcapa[:]=1
         case _ :
            exit('rho,eta - unknown experiment')  

    print("     -> swarm_rho (m,M) %.5e %.5e " %(np.min(swarm_rho),np.max(swarm_rho)))
    print("     -> swarm_eta (m,M) %.5e %.5e " %(np.min(swarm_eta),np.max(swarm_eta)))

    if debug_ascii: np.savetxt('swarm_rho.ascii',np.array([swarm_x,swarm_y,swarm_rho]).T,header='# x,y,rho')
    if debug_ascii: np.savetxt('swarm_eta.ascii',np.array([swarm_x,swarm_y,swarm_eta]).T,header='# x,y,eta')

    print("compute rho,eta on particles: %.3fs" % (clock.time()-start)) ; timings[15]+=clock.time()-start

    ###########################################################################
    # project particle properties on elements 
    ###########################################################################
    start=clock.time()

    rho_elemental,eta_elemental,nparticle_elemental=\
    project_particles_on_elements(nel,nparticle,swarm_rho,swarm_eta,swarm_iel,averaging)

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
    ###########################################################################
    start=clock.time()

    rhoq=Q1_project_nodal_field_onto_qpoints(rho_nodal,nqel,nel,N_P,icon_V)
    etaq=Q1_project_nodal_field_onto_qpoints(eta_nodal,nqel,nel,N_P,icon_V)

    if solve_T:
       Tq=Q2_project_nodal_field_onto_qpoints(T,nqel,nel,N_V,icon_V)
       hcapaq=Q1_project_nodal_field_onto_qpoints(hcapa_nodal,nqel,nel,N_P,icon_V)
       hcondq=Q1_project_nodal_field_onto_qpoints(hcond_nodal,nqel,nel,N_P,icon_V)

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

    B=np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
    N_mat=np.zeros((3,m_P),dtype=np.float64) # matrix  
    rhs=np.zeros(Nfem,dtype=np.float64)     # right hand side of Ax=b

    counter=0
    for iel in range(0,nel):

        f_el=np.zeros((ndof_V_el),dtype=np.float64)
        K_el=np.zeros((ndof_V_el,ndof_V_el),dtype=np.float64)
        G_el=np.zeros((ndof_V_el,m_P),dtype=np.float64)
        h_el=np.zeros((m_P),dtype=np.float64)

        for iq in range(0,nqel):

            JxWq=jcob*weightq[iq]

            for i in range(0,m_V):
                dNdx=dNdx_V[iq,i] 
                dNdy=dNdy_V[iq,i] 
                B[0,2*i  ]=dNdx
                B[1,2*i+1]=dNdy
                B[2,2*i  ]=dNdy
                B[2,2*i+1]=dNdx

            K_el+=B.T.dot(C.dot(B))*etaq[iel,iq]*JxWq

            for i in range(0,m_V):
                f_el[ndof_V*i+1]+=N_V[iq,i]*JxWq*rhoq[iel,iq]*gy

            N_mat[0,0:m_P]=N_P[iq,0:m_P]
            N_mat[1,0:m_P]=N_P[iq,0:m_P]

            G_el-=B.T.dot(N_mat)*JxWq

        # end for iq

        G_el*=eta_ref/Lx

        # impose b.c. 
        for ikk in range(0,ndof_V_el):
            m1=local_to_globalV[ikk,iel]
            if bc_fix_V[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,ndof_V_el):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
               K_el[ikk,:]=0
               K_el[:,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val_V[m1]
               h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
               G_el[ikk,:]=0

        # assemble matrix and right hand side
        for ikk in range(ndof_V_el):
            m1=local_to_globalV[ikk,iel]
            for jkk in range(ndof_V_el):
                VV_V[counter]=K_el[ikk,jkk]
                counter+=1
            for jkk in range(0,m_P):
                VV_V[counter]=G_el[ikk,jkk]
                counter+=1
                VV_V[counter]=G_el[ikk,jkk]
                counter+=1
            rhs[m1]+=f_el[ikk]
        for k2 in range(0,m_P):
            m2=icon_P[k2,iel]
            rhs[Nfem_V+m2]+=h_el[k2]

    if debug_nan and np.isnan(np.sum(II_V)): exit('nan found in II_V')
    if debug_nan and np.isnan(np.sum(JJ_V)): exit('nan found in JJ_V')
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
    p=sol[Nfem_V:Nfem]*(eta_ref/Lx)

    if debug_nan and np.isnan(np.sum(u)): exit('nan found in u')
    if debug_nan and np.isnan(np.sum(v)): exit('nan found in v')
    if debug_nan and np.isnan(np.sum(p)): exit('nan found in p')

    print("     -> u (m,M) %.3e %.3e %s" %(np.min(u)/vel_scale,np.max(u)/vel_scale,vel_unit))
    print("     -> v (m,M) %.3e %.3e %s" %(np.min(v)/vel_scale,np.max(v)/vel_scale,vel_unit))
    print("     -> p (m,M) %.3e %.3e %s" %(np.min(p)/p_scale,np.max(p)/p_scale,p_unit))

    vstats_file.write("%.3e %.3e %.3e %.3e %.3e\n" % (istep,np.min(u)/vel_scale,np.max(u)/vel_scale,\
                                                            np.min(u)/vel_scale,np.max(u)/vel_scale))

    if debug_ascii: np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
    if debug_ascii: np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

    print("split vel into u,v: %.3f s" % (clock.time()-start)) ; timings[14]+=clock.time()-start

    ###########################################################################
    # compute timestep
    ###########################################################################
    start=clock.time()

    dt1=CFLnb*(Lx/nelx)/np.max(np.sqrt(u**2+v**2))
    print('     -> dt1= %.3e %s' %(dt1/time_scale,time_unit))
    
    if solve_T:
       dt2=CFLnb*(Lx/nelx)**2/(hcond/hcapa/rho0)
       print('     -> dt2= %.3e %s' %(dt2/time_scale,time_unit))
    else:
       dt2=1e50
    dt=np.min([dt1,dt2])

    geological_time+=dt

    print('     -> dt = %.3e %s' %(dt/time_scale,time_unit))
    print('     -> geological time = %e %s' %(geological_time/time_scale,time_unit))

    dt_file.write("%e %e %e %e\n" % (geological_time,dt1,dt2,dt)) ; dt_file.flush()

    print("compute time step: %.3f s" % (clock.time()-start)) ; timings[19]+=clock.time()-start

    ###########################################################################
    # normalise pressure: simple approach to have <p> @ surface = 0
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

    print("     -> p (m,M) %.3e %.3e %s" %(np.min(p),np.max(p),p_unit))

    pstats_file.write("%d %.3e %.3e\n" % (istep,np.min(p),np.max(p)))

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

       #VV_T,rhs=build_matrix_T(nel,nqel,m_V,u,v,T,N_V,dNdx_V,dNdy_V,\
       #                             icon_V,jcob,weightq,bc_fix_T,bc_val_T

       Tvect=np.zeros(m_T,dtype=np.float64)   
       rhs=np.zeros(Nfem_T,dtype=np.float64)    # FE rhs 
       B=np.zeros((2,m_T),dtype=np.float64)     # gradient matrix B 
       N_mat=np.zeros((m_T,1),dtype=np.float64)   # shape functions

       counter=0
       for iel in range(0,nel):

           b_el=np.zeros(m_T,dtype=np.float64)
           A_el=np.zeros((m_T,m_T),dtype=np.float64)
           Ka=np.zeros((m_T,m_T),dtype=np.float64)   # elemental advection matrix 
           Kd=np.zeros((m_T,m_T),dtype=np.float64)   # elemental diffusion matrix 
           MM=np.zeros((m_T,m_T),dtype=np.float64)   # elemental mass matrix 
           velq=np.zeros((1,ndim),dtype=np.float64)

           Tvect[0:m_T]=T[icon_V[0:m_T,iel]]

           for iq in range(0,nqel):

               JxWq=jcob*weightq[iq]

               N=N_V[iq,:]

               velq[0,0]=uq[iel,iq]
               velq[0,1]=vq[iel,iq]

               B[0,:]=dNdx_V[iq,:]
               B[1,:]=dNdy_V[iq,:]
               
               MM+=np.outer(N,N)*rho0*hcapa*JxWq # mass matrix
   
               Kd+=B.T.dot(B)*hcond*JxWq # diffusion matrix
               
               Ka+=np.outer(N,velq.dot(B))*rho0*hcapa*JxWq # advection matrix

               if EBA:
                  Tq=np.dot(N_V[iq,:],T[icon_V[:,iel]])
                  exxq=np.dot(dNdx_V[iq,:],u[icon_V[:,iel]])
                  eyyq=np.dot(dNdy_V[iq,:],v[icon_V[:,iel]])
                  exyq=np.dot(dNdy_V[iq,:],u[icon_V[:,iel]])*0.5\
                      +np.dot(dNdx_V[iq,:],v[icon_V[:,iel]])*0.5
                  dpdxq=np.dot(dNdx_V[iq,:],q[icon_V[:,iel]])
                  dpdyq=np.dot(dNdy_V[iq,:],q[icon_V[:,iel]])
                  #viscous dissipation
                  b_el[:]+=N[:]*JxWq*2*eta(Tq,xq,yq,eta0)*(exxq**2+eyyq**2+2*exyq**2) 
                  #adiabatic heating
                  b_el[:]+=N[:]*JxWq*alphaT*Tq*(velq[0,0]*dpdxq+velq[0,1]*dpdyq)  
   
           #end for

           A_el+=MM+(Ka+Kd)*dt*0.5
           b_el+=(MM-(Ka+Kd)*dt*0.5).dot(Tvect)

           # apply boundary conditions
           for k1 in range(0,m_V):
               m1=icon_V[k1,iel]
               if bc_fix_T[m1]:
                  Aref=A_el[k1,k1]
                  for k2 in range(0,m_V):
                      m2=icon_V[k2,iel]
                      b_el[k2]-=A_el[k2,k1]*bc_val_T[m1]
                      A_el[k1,k2]=0
                      A_el[k2,k1]=0
                  #end for
                  A_el[k1,k1]=Aref
                  b_el[k1]=Aref*bc_val_T[m1]
               #end for
           #end for

           # assemble matrix K_mat and right hand side rhs
           for ikk in range(m_T):
               m1=icon_V[ikk,iel]
               for jkk in range(m_T):
                   VV_T[counter]=A_el[ikk,jkk]
                   counter+=1
               rhs[m1]+=b_el[ikk]
           #end for

       #end for iel

       print("build FE matrix : %.3f s" % (clock.time()-start)) ; timings[4]+=clock.time()-start

       ###########################################################################
       # solve system
       ###########################################################################
       start = clock.time()

       sparse_matrix=sparse.coo_matrix((VV_T,(II_T,JJ_T)),shape=(Nfem_T,Nfem_T)).tocsr()

       T=sps.linalg.spsolve(sparse_matrix,rhs)

       if debug_nan and np.isnan(np.sum(T)): exit('nan found in T')

       print("     T (m,M) %.3e %.3e " %(np.min(T),np.max(T)))

       if debug_ascii: np.savetxt('T.ascii',np.array([x_V,y_V,T]).T,header='# x,y,T')

       print("solve T time: %.3f s" % (clock.time()-start)) ; timings[5]+=clock.time()-start

    #end if solve_T

    ###########################################################################
    # compute vrms 
    ###########################################################################
    start=clock.time()

    vrms=compute_vrms(nel,nqel,weightq,icon_V,u,v,N_V,Lx,Ly,jcob)

    vrms_file.write("%e %e \n" % (geological_time/time_scale,vrms/vel_scale)) ; vrms_file.flush()

    print("     istep= %.6d ; vrms   = %.3e %s" %(istep,vrms/vel_scale,vel_unit))

    print("compute vrms: %.3f s" % (clock.time()-start)) ; timings[6]+=clock.time()-start

    ###########################################################################
    # compute nodal heat flux 
    ###########################################################################
    start=clock.time()

    if solve_T: 
       qx_nodal,qy_nodal=compute_nodal_heat_flux(icon_V,T,hcond,nn_V,m_V,nel,dNdx_V_n,dNdy_V_n)

       print("     -> qx_nodal (m,M) %.3e %.3e " %(np.min(qx_nodal),np.max(qx_nodal)))
       print("     -> qy_nodal (m,M) %.3e %.3e " %(np.min(qy_nodal),np.max(qy_nodal)))

    else:
       qx_nodal=0 
       qy_nodal=0 

    print("compute nodal heat flux: %.3f s" % (clock.time()-start)) ; timings[7]+=clock.time()-start

    ###########################################################################
    # compute Nusselt number at top
    ###########################################################################
    start=clock.time()

    if istep%every_Nu==0 and solve_T: 

       qy_top=0
       qy_bot=0
       Nusselt=0
       for iel in range(0,nel):
           if y_V[icon_V[m_V-1,iel]]/Ly>1-eps: # top row of nodes 
              sq=+1
              for iq in range(0,nqperdim):
                  rq=qcoords[iq]
                  N=basis_functions_V(rq,sq)
                  q_y=np.dot(N,qy_nodal[icon_V[:,iel]])
                  Nusselt+=q_y*(hx/2)*qweights[iq]
                  qy_top+=q_y*(hx/2)*qweights[iq]
              #end for
           #end if
           if y_V[icon_V[0,iel]]/Ly<eps: # bottom row of nodes
              sq=-1
              for iq in range(0,nqperdim):
                  rq=qcoords[iq]
                  N=basis_functions_V(rq,sq)
                  q_y=np.dot(N,qy_nodal[icon_V[:,iel]])
                  qy_bot-=q_y*(hx/2)*qweights[iq]
           #end if
       #end for

       Nusselt=np.abs(Nusselt)/Lx

       print("     -> qy_bot,qy_top= %.3e %.3e " %(qy_bot,qy_top))
       print("     -> Nusselt= %.2e " %(Nusselt))

       Nu_file.write("%e %e \n" % (geological_time/time_scale,Nusselt)) ; Nu_file.flush()

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

       print("compute T profile: %.3f s" % (clock.time() - start)) ; timings[9]+=clock.time()-start

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
    # plot of solution
    ###########################################################################
    start=clock.time()

    if istep%every_vtu==0: 
       export_solution_to_vtu(istep,nel,nn_V,m_V,solve_T,vel_scale,TKelvin,x_V,y_V,u,v,q,T,
                              eta_nodal,rho_nodal,exx_nodal,eyy_nodal,exy_nodal,qx_nodal,qy_nodal,
                              rho_elemental,eta_elemental,nparticle_elemental,icon_V)

       print("export solution to vtu file: %.3f s" % (clock.time()-start)) ; timings[10]+=clock.time()-start

    ########################################################################
    # export particles to vtu file
    ########################################################################
    start=clock.time()

    if istep%every_vtu==0 or istep==nstep-1: 
       export_swarm_to_vtu(istep,nparticle,solve_T,vel_scale,swarm_x,swarm_y,\
                           swarm_u,swarm_v,swarm_mat,swarm_rho,swarm_eta,\
                           swarm_paint,swarm_exx,swarm_eyy,swarm_exy,swarm_T,\
                           swarm_hcond,swarm_hcapa) 

       print("export particles to vtu file: %.3f s" % (clock.time()-start)) ; timings[20]+=clock.time()-start

    ########################################################################
    # export quadrature points to vtu file
    ########################################################################
    start=clock.time()

    if istep%every_vtu==0 or istep==nstep-1: 
       export_quadpoints_to_vtu(istep,nel,nqel,nq,xq,yq,rhoq,etaq)

       print("export quad pts to vtu file: %.3f s" % (clock.time()-start)) ; timings[22]+=clock.time()-start

    ###########################################################################

    u_mem=u.copy()
    v_mem=v.copy()
    T_mem=T.copy()

    ###########################################################################

    if istep%10==0 or istep==nstep-1:

       duration=clock.time()-topstart

       print("----------------------------------------------------------------------")
       print("build FE matrix V: %8.3f s      (%.3f s per call) | %5.2f percent" % (timings[1],timings[1]/(istep+1),timings[1]/duration*100)) 
       print("solve system V: %8.3f s         (%.3f s per call) | %5.2f percent" % (timings[2],timings[2]/(istep+1),timings[2]/duration*100))
       print("build matrix T: %8.3f s         (%.3f s per call) | %5.2f percent" % (timings[4],timings[4]/(istep+1),timings[4]/duration*100))
       print("solve system T: %8.3f s         (%.3f s per call) | %5.2f percent" % (timings[5],timings[5]/(istep+1),timings[5]/duration*100))
       print("comp. vrms: %8.3f s             (%.3f s per call) | %5.2f percent" % (timings[6],timings[6]/(istep+1),timings[6]/duration*100))
       print("comp. nodal p: %8.3f s          (%.3f s per call) | %5.2f percent" % (timings[3],timings[3]/(istep+1),timings[3]/duration*100))
       print("comp. nodal sr: %8.3f s         (%.3f s per call) | %5.2f percent" % (timings[11],timings[11]/(istep+1),timings[11]/duration*100))
       print("comp. nodal heat flux: %8.3f s  (%.3f s per call) | %5.2f percent" % (timings[7],timings[7]/(istep+1),timings[7]/duration*100))
       print("comp. T profile: %8.3f s        (%.3f s per call) | %5.2f percent" % (timings[9],timings[9]/(istep+1),timings[9]/duration*100)) 
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
       print("----------------------------------------------------------------------")
       print("compute time per timestep: %.3e" %(duration/(istep+1)))
       print("----------------------------------------------------------------------")

#end for istep

###############################################################################
# close files
###############################################################################
       
vstats_file.close()
pstats_file.close()
vrms_file.close()
dt_file.close()
Nu_file.close()

###############################################################################

print("-----------------------------")
print("total compute time: %.1f s" % (duration))
print("sum timings: %.1f s" % (np.sum(timings)))
print("-----------------------------")
    
###############################################################################
