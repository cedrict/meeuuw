###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numba
import numpy as np

###################################################################################################


@numba.njit
def project_nodal_Pfield_onto_qpoints(phi_nodal, nq_per_element, nel, m_P, N_P, icon_P):
    """
    Args:
    Returns:
    """

    phiq = np.zeros((nel, nq_per_element), dtype=np.float64)

    for iel in range(0, nel):
        for iq in range(0, nq_per_element):
            phiq[iel, iq] = np.dot(N_P[iq, 0:m_P], phi_nodal[icon_P[0:m_P, iel]])

    return phiq


###################################################################################################


@numba.njit
def project_nodal_Vfield_onto_qpoints(phi_nodal, nq_per_element, nel, m_V, N_V, icon_V):
    """
    Args:
    Returns:
    """

    phiq = np.zeros((nel, nq_per_element), dtype=np.float64)

    for iel in range(0, nel):
        for iq in range(0, nq_per_element):
            phiq[iel, iq] = np.dot(N_V[iq, 0:m_V], phi_nodal[icon_V[0:m_V, iel]])

    return phiq


###################################################################################################


@numba.njit
def project_nodal_Tfield_onto_qpoints(phi_nodal, nq_per_element, nel, m_T, N_T, icon_T):
    """
    Args:
    Returns:
    """

    phiq = np.zeros((nel, nq_per_element), dtype=np.float64)

    for iel in range(0, nel):
        for iq in range(0, nq_per_element):
            phiq[iel, iq] = np.dot(N_T[iq, 0:m_T], phi_nodal[icon_T[0:m_T, iel]])

    return phiq


###################################################################################################
