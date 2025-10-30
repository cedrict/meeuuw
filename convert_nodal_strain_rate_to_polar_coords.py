import numpy as np 
import numba

###############################################################################
#def convert_nodal_strain_rate_to_polar_coords(theta,Txx,Tyy,Txy):

@numba.njit
def convert_tensor_to_polar_coords(theta,Txx,Tyy,Txy):

    Trr=Txx*(np.cos(theta))**2\
       +2*Txy*np.sin(theta)*np.cos(theta)\
       +Tyy*(np.sin(theta))**2
    Ttt=Txx*(np.sin(theta))**2\
       -2*Txy*np.sin(theta)*np.cos(theta)\
       +Tyy*(np.cos(theta))**2
    Trt=(Tyy-Txx)*np.sin(theta)*np.cos(theta)\
       +Txy*((np.cos(theta))**2-(np.sin(theta))**2)

    e=np.sqrt(0.5*(Trr**2+Ttt**2)+Trt**2)

    return Trr,Ttt,Trt,e

###############################################################################
