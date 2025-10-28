import numpy as np 
import numba

###############################################################################

@numba.njit
def convert_nodal_strain_rate_to_polar_coords(theta,exx,eyy,exy):

    #print(np.cos(theta))
    #print(np.sin(theta))

    err=exx*(np.cos(theta))**2\
       +2*exy*np.sin(theta)*np.cos(theta)\
       +eyy*(np.sin(theta))**2
    ett=exx*(np.sin(theta))**2\
       -2*exy*np.sin(theta)*np.cos(theta)\
       +eyy*(np.cos(theta))**2
    ert=(eyy-exx)*np.sin(theta)*np.cos(theta)\
       +exy*((np.cos(theta))**2-(np.sin(theta))**2)

    e=np.sqrt(0.5*(err**2+ett**2)+ert**2)

    return err,ett,ert,e

###############################################################################
