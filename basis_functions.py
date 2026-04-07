###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np
import numba

###################################################################################################
# Q2 basis functions
###################################################################################################

@numba.njit
def basis_functions_V(r,s):
    N0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
    N1= 0.5*r*(r+1.) * 0.5*s*(s-1.)
    N2= 0.5*r*(r+1.) * 0.5*s*(s+1.)
    N3= 0.5*r*(r-1.) * 0.5*s*(s+1.)
    N4=    (1.-r**2) * 0.5*s*(s-1.)
    N5= 0.5*r*(r+1.) *    (1.-s**2)
    N6=    (1.-r**2) * 0.5*s*(s+1.)
    N7= 0.5*r*(r-1.) *    (1.-s**2)
    N8=    (1.-r**2) *    (1.-s**2)
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7,N8],dtype=np.float64)

@numba.njit
def basis_functions_V_dr(r,s):
    dNdr0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
    dNdr1= 0.5*(2.*r+1.) * 0.5*s*(s-1)
    dNdr2= 0.5*(2.*r+1.) * 0.5*s*(s+1)
    dNdr3= 0.5*(2.*r-1.) * 0.5*s*(s+1)
    dNdr4=       (-2.*r) * 0.5*s*(s-1)
    dNdr5= 0.5*(2.*r+1.) *   (1.-s**2)
    dNdr6=       (-2.*r) * 0.5*s*(s+1)
    dNdr7= 0.5*(2.*r-1.) *   (1.-s**2)
    dNdr8=       (-2.*r) *   (1.-s**2)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7,dNdr8],dtype=np.float64)

@numba.njit
def basis_functions_V_dt(r,s):
    dNdt0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
    dNdt1= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
    dNdt2= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
    dNdt3= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
    dNdt4=    (1.-r**2) * 0.5*(2.*s-1.)
    dNdt5= 0.5*r*(r+1.) *       (-2.*s)
    dNdt6=    (1.-r**2) * 0.5*(2.*s+1.)
    dNdt7= 0.5*r*(r-1.) *       (-2.*s)
    dNdt8=    (1.-r**2) *       (-2.*s)
    return np.array([dNdt0,dNdt1,dNdt2,dNdt3,dNdt4,dNdt5,dNdt6,dNdt7,dNdt8],dtype=np.float64)

###################################################################################################
# Q1 basis functions
###################################################################################################

@numba.njit
def basis_functions_P(r,s):
    N0=0.25*(1-r)*(1-s)
    N1=0.25*(1+r)*(1-s)
    N2=0.25*(1+r)*(1+s)
    N3=0.25*(1-r)*(1+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

###################################################################################################

@numba.njit
def basis_functions_P_dr(r,s):
    dNdr0=-0.25*(1-s)
    dNdr1= 0.25*(1-s)
    dNdr2= 0.25*(1+s)
    dNdr3=-0.25*(1+s)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)

@numba.njit
def basis_functions_P_dt(r,s):
    dNdt0=-0.25*(1-r)
    dNdt1=-0.25*(1+r)
    dNdt2= 0.25*(1+r)
    dNdt3= 0.25*(1-r)
    return np.array([dNdt0,dNdt1,dNdt2,dNdt3],dtype=np.float64)

###################################################################################################
