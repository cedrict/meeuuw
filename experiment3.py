import numpy as np
import numba

###############################################################################

def assign_parameters(icase):
    Ra=1e2
    match(icase):
         case 1 :
             sigma_y=1.
             gamma_y=np.log(1.)
         case 2 :
             sigma_y = 1
             gamma_y=np.log(1.)
         case 3 :
             sigma_y = 1
             gamma_y=np.log(10.)
         case 4 :
             sigma_y = 1
             gamma_y=np.log(10.)
         case 5 :
             sigma_y=4.
             gamma_y=np.log(10.)
         case _ :
             exit('pb in assign_parameters')
    return Ra,sigma_y,gamma_y

###############################################################################

Lx=1
Ly=1
eta_ref=1
solve_T=True
vel_scale=1 ; vel_unit=' '
time_scale=1 ; time_unit=' '
p_scale=1 ; p_unit=' '
Ttop=0
Tbottom=1
alphaT=1e-4
hcond=1  
hcapa=1 
rho0=1
Ra=1e4
gy=-Ra/alphaT 
TKelvin=0
pressure_normalisation='surface'
every_Nu=1
end_time=0.25
case_tosi=1
gamma_T=np.log(1e5)
eta_star=1e-3 
eta_ref=1e-2
Ra,sigma_y,gamma_y=assign_parameters(case_tosi)
eta_min=1e-5
eta_max=1

nelx=32
nely=32
nstep=10

CFLnb=0.5

###############################################################################

@numba.njit
def viscosity(T,exx,eyy,exy,y,gamma_T,gamma_y,sigma_y,eta_star,icase):
    #-------------------
    # tosi et al, case 1
    #-------------------
    if icase==1:
       val=np.exp(-gamma_T*T)
    #-------------------
    # tosi et al, case 2
    #-------------------
    elif icase==2:
       e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
       e=max(e,1e-12)
       eta_lin=np.exp(-gamma_T*T)
       eta_plast=eta_star + sigma_y/(np.sqrt(2.)*e)
       val=2./(1./eta_lin + 1./eta_plast)
    #-------------------
    # tosi et al, case 3
    #-------------------
    elif icase==3:
       val=np.exp(-gamma_T*T+gamma_y*(1-y))
    #-------------------
    # tosi et al, case 4
    #-------------------
    elif icase==4:
       e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
       eta_lin=np.exp(-gamma_T*T+gamma_y*(1-y))
       eta_plast=eta_star + sigma_y/(np.sqrt(2)*e)
       val=2/(1/eta_lin + 1/eta_plast)
    #-------------------
    # tosi et al, case 5
    #-------------------
    elif icase==5:
       e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
       eta_lin=np.exp(-gamma_T*T+gamma_y*(1-y))
       eta_plast=eta_star + sigma_y/(np.sqrt(2)*e)
       val=2/(1/eta_lin + 1/eta_plast)
    val=min(2.0,val)
    val=max(1.e-5,val)
    return val

###############################################################################
