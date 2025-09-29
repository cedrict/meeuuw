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
