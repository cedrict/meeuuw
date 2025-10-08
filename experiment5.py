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
vel_scale=1 ; vel_unit=' '
time_scale=1 ; time_unit=' '
p_scale=1 ; p_unit=' '
Ttop=0
Tbottom=1

alphaT=1e-4
T0=0
hcond0=1   
hcapa0=1  
rho0=1
Ra_surf=100
gy=-Ra_surf/alphaT   *2 #visc 
sigma_y=2

TKelvin=0
pressure_normalisation='surface'
every_Nu=1
end_time=0.25
gamma_T=np.log(1e5)
eta_star=1e-5 
eta_ref=1e-2
           
nelx=128
nely=int(Ly/Lx*nelx)
nstep=10000

CFLnb=0.5


@numba.njit
def viscosity(T,exx,eyy,exy,y):
    e=max(np.sqrt(exx**2+eyy**2+2*exy**2),1e-10)
    eta_T=np.exp(-gamma_T*T)
    eta_e=eta_star + sigma_y/e
    val=2/(1/eta_T + 1/eta_e)
    val=min(2.0,val)
    val=max(1.e-5,val)
    return val

def initial_temperature(x,y):
    return (Tbottom-Ttop)*(Ly-y)/Ly+Ttop +0.01*np.cos(3*np.pi*x/Lx)*np.sin(3*np.pi*y/Ly)



