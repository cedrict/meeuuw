import numpy as np
import numba

###############################################################################

@numba.njit
def Q1_project_nodal_field_onto_qpoints(phi_nodal,nqel,nel,N_P,icon_V):

    phiq=np.zeros((nel,nqel),dtype=np.float64)

    for iel in range(0,nel):
        for iq in range(0,nqel):
            phiq[iel,iq]=np.dot(N_P[iq,0:4],phi_nodal[icon_V[0:4,iel]])

    return phiq

###############################################################################

@numba.njit
def Q2_project_nodal_field_onto_qpoints(phi_nodal,nqel,nel,N_V,icon_V):

    phiq=np.zeros((nel,nqel),dtype=np.float64)

    for iel in range(0,nel):
        for iq in range(0,nqel):
            phiq[iel,iq]=np.dot(N_V[iq,0:9],phi_nodal[icon_V[0:9,iel]])

    return phiq

###############################################################################
