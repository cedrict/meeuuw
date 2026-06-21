###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

from basis_functions import (
    basis_functions_P,
    basis_functions_P_dr,
    basis_functions_P_dt,
    basis_functions_V,
    basis_functions_V_dr,
    basis_functions_V_dt,
    basis_functions_T,
    basis_functions_T_dr,
    basis_functions_T_dt,
)

###################################################################################################
# An underlying assumption here is that all elements are straight edged so that 
# all basis functions (Q1, Q2, Q1+) are suitable to compute the jacobians.

def basis_functions_setup_q(nq_per_dim, m_V, m_P, m_T, nel, x_V, z_V, icon_V, qcoords, qweights, volume):

    nq_per_element = nq_per_dim**2

    jcb = np.zeros((2, 2), dtype=np.float64)
    rq = np.zeros(nq_per_element, dtype=np.float64)
    tq = np.zeros(nq_per_element, dtype=np.float64)
    weightq = np.zeros(nq_per_element, dtype=np.float64)
    N_V = np.zeros((nq_per_element, m_V), dtype=np.float64)
    N_P = np.zeros((nq_per_element, m_P), dtype=np.float64)
    N_T = np.zeros((nq_per_element, m_T), dtype=np.float64)
    dNdr_V = np.zeros((nq_per_element, m_V), dtype=np.float64)
    dNdt_V = np.zeros((nq_per_element, m_V), dtype=np.float64)
    dNdr_P = np.zeros((nq_per_element, m_P), dtype=np.float64)
    dNdt_P = np.zeros((nq_per_element, m_P), dtype=np.float64)
    dNdr_T = np.zeros((nq_per_element, m_T), dtype=np.float64)
    dNdt_T = np.zeros((nq_per_element, m_T), dtype=np.float64)
    area = np.zeros(nel, dtype=np.float64)
    JxWq = np.zeros((nel, nq_per_element), dtype=np.float64)
    jcbi00q = np.zeros((nel, nq_per_element), dtype=np.float64)
    jcbi01q = np.zeros((nel, nq_per_element), dtype=np.float64)
    jcbi10q = np.zeros((nel, nq_per_element), dtype=np.float64)
    jcbi11q = np.zeros((nel, nq_per_element), dtype=np.float64)

    for iel in range(0, nel):
        cq = 0
        for iq in range(0, nq_per_dim):
            for jq in range(0, nq_per_dim):
                if iel == 0:
                    rq[cq] = qcoords[iq]
                    tq[cq] = qcoords[jq]
                    weightq[cq] = qweights[iq] * qweights[jq]
                    N_V[cq, 0:m_V] = basis_functions_V(rq[cq], tq[cq])
                    N_P[cq, 0:m_P] = basis_functions_P(rq[cq], tq[cq])
                    N_T[cq, 0:m_T] = basis_functions_T(rq[cq], tq[cq])
                    dNdr_V[cq, 0:m_V] = basis_functions_V_dr(rq[cq], tq[cq])
                    dNdt_V[cq, 0:m_V] = basis_functions_V_dt(rq[cq], tq[cq])
                    dNdr_P[cq, 0:m_P] = basis_functions_P_dr(rq[cq], tq[cq])
                    dNdt_P[cq, 0:m_P] = basis_functions_P_dt(rq[cq], tq[cq])
                    dNdr_T[cq, 0:m_T] = basis_functions_T_dr(rq[cq], tq[cq])
                    dNdt_T[cq, 0:m_T] = basis_functions_T_dt(rq[cq], tq[cq])
                # end if
                jcb[0, 0] = np.dot(dNdr_V[cq, :], x_V[icon_V[:, iel]])
                jcb[0, 1] = np.dot(dNdr_V[cq, :], z_V[icon_V[:, iel]])
                jcb[1, 0] = np.dot(dNdt_V[cq, :], x_V[icon_V[:, iel]])
                jcb[1, 1] = np.dot(dNdt_V[cq, :], z_V[icon_V[:, iel]])
                jcbi = np.linalg.inv(jcb)
                JxWq[iel, cq] = np.linalg.det(jcb) * weightq[cq]
                jcbi00q[iel, cq] = jcbi[0, 0]
                jcbi01q[iel, cq] = jcbi[0, 1]
                jcbi10q[iel, cq] = jcbi[1, 0]
                jcbi11q[iel, cq] = jcbi[1, 1]
                area[iel] += JxWq[iel, cq]
                cq += 1
            # end for
        # end for
    # end for

    print("     -> area (m,M) %.4e %.4e " % (np.min(area), np.max(area)))
    print("     -> total area %.4e %.4e " % (area.sum(), volume))

    return (
        rq,
        tq,
        weightq,
        N_V,
        N_P,
        N_T,
        dNdr_V,
        dNdt_V,
        dNdr_P,
        dNdt_P,
        dNdr_T,
        dNdt_T,
        JxWq,
        jcbi00q,
        jcbi01q,
        jcbi10q,
        jcbi11q,
        area,
    )


###################################################################################################
