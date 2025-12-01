import numpy as np
import numba

###############################################################################

@numba.njit
def effective(Txx,Tzz,Txz):
    return np.sqrt(0.5*(Txx**2+Tzz**2)+Txz**2)

###############################################################################

@numba.njit
def convert_tensor_to_polar_coords(theta,Txx,Tzz,Txz):

    Trr=Txx*(np.cos(theta))**2 +2*Txz*np.sin(theta)*np.cos(theta) +Tzz*(np.sin(theta))**2

    Ttt=Txx*(np.sin(theta))**2 -2*Txz*np.sin(theta)*np.cos(theta) +Tzz*(np.cos(theta))**2

    Trt=(Tzz-Txx)*np.sin(theta)*np.cos(theta) +Txz*((np.cos(theta))**2-(np.sin(theta))**2)

    return Trr,Ttt,Trt

###############################################################################

@numba.njit
def convert_tensor_to_spherical_coords(theta_polar,Txx,Tzz,Txz):

    theta_sph=np.pi/2-theta_polar

    sin_theta=np.sin(theta_sph)
    cos_theta=np.cos(theta_sph)

    #Trr=Txx*sin_theta**2 +2*Txz*sin_theta*cos_theta +Tzz*cos_theta**2
    #Ttt=Txx*cos_theta**2 -2*Txz*sin_theta*cos_theta +Tzz*sin_theta**2
    #Trt=(Txx-Tzz)*sin_theta*cos_theta +Txz*(cos_theta**2-sin_theta**2)

    sin_twotheta=np.sin(2.*theta_sph)
    cos_twotheta=np.cos(2.*theta_sph)
    Trr=Txx*sin_theta**2 +Txz*sin_twotheta +Tzz*cos_theta**2
    Ttt=Txx*cos_theta**2 -Txz*sin_twotheta +Tzz*sin_theta**2
    Trt=0.5*(Txx-Tzz)*sin_twotheta +Txz*cos_twotheta

    return Trr,Ttt,Trt

###############################################################################
