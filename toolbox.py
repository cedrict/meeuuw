###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np
import numba

###################################################################################################

@numba.njit
def effective(Txx,Tzz,Txz):
    return np.sqrt(0.5*(Txx**2+Tzz**2)+Txz**2)

###################################################################################################

@numba.njit
def convert_tensor_to_polar_coords(theta,Txx,Tzz,Txz):

    Trr=Txx*(np.cos(theta))**2 +2*Txz*np.sin(theta)*np.cos(theta) +Tzz*(np.sin(theta))**2

    Ttt=Txx*(np.sin(theta))**2 -2*Txz*np.sin(theta)*np.cos(theta) +Tzz*(np.cos(theta))**2

    Trt=(Tzz-Txx)*np.sin(theta)*np.cos(theta) +Txz*((np.cos(theta))**2-(np.sin(theta))**2)

    return Trr,Ttt,Trt

###################################################################################################

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

###################################################################################################

def inspect_element(iel,m_V,icon_V,x_V,z_V,rho_n,eta_n,nq_per_element,xq,zq,rhoq,etaq):
    for k in range(0,m_V):
        knode=icon_V[k,iel]
        print(x_V[knode],z_V[knode],eta_n[knode],rho_n[knode])
    for iq in range(0,nq_per_element):
        print(xq[iel,iq],zq[iel,iq],etaq[iel,iq],rhoq[iel,iq])

###################################################################################################
# this functions exports the values of the solution fields at a given set of user-chosen 
# locations, provided these locations correspond to a V node location.
# A more versatile algorithm could/should be implemented in the future.
###################################################################################################

@numba.njit
def sample_solution_box(nn_V,x_V,z_V,u,w,q,T,nsamplepoints,xsamplepoints,zsamplepoints,Lx,Lz,nelx,nelz):
    """
    Args:
     nn_V: number of V nodes
     x_V,z_V: coordinates of V nodes
     u,w,q,T: fields on V nodes
     nsamplepoints: nb of sampling points
     xsamplepoints,zsamplepoints: coordinates of sampling points
     Lx,Lz: domain size
     nelx,nelz: number of elements
    Returns:
     -
    """

    for isp in range(0,nsamplepoints):
        for i in range(nn_V):
            if abs(x_V[i]-xsamplepoints[isp])/Lx<1e-6 and\
               abs(z_V[i]-zsamplepoints[isp])/Lz<1e-6:
               print('sample ->',x_V[i],z_V[i],u[i],w[i],q[i],T[i],nelx,nelz)

###################################################################################################
