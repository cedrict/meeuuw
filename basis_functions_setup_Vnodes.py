###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

from basis_functions import basis_functions_P, basis_functions_V_dr, basis_functions_V_dt

###################################################################################################


def basis_functions_setup_Vnodes(m_V, m_P, nel, r_V, t_V, x_V, z_V, icon_V):

    jcb = np.zeros((2, 2), dtype=np.float64)
    N_P_n = np.zeros((m_V, m_P), dtype=np.float64)
    dNdr_V_n = np.zeros((m_V, m_V), dtype=np.float64)
    dNdt_V_n = np.zeros((m_V, m_V), dtype=np.float64)
    jcbi00n = np.zeros((nel, m_V), dtype=np.float64)
    jcbi01n = np.zeros((nel, m_V), dtype=np.float64)
    jcbi10n = np.zeros((nel, m_V), dtype=np.float64)
    jcbi11n = np.zeros((nel, m_V), dtype=np.float64)

    for iel in range(0, nel):
        for i in range(0, m_V):
            if iel == 0:
                N_P_n[i, 0:m_P] = basis_functions_P(r_V[i], t_V[i])
                dNdr_V_n[i, 0:m_V] = basis_functions_V_dr(r_V[i], t_V[i])
                dNdt_V_n[i, 0:m_V] = basis_functions_V_dt(r_V[i], t_V[i])
            jcb[0, 0] = np.dot(dNdr_V_n[i, :], x_V[icon_V[:, iel]])
            jcb[0, 1] = np.dot(dNdr_V_n[i, :], z_V[icon_V[:, iel]])
            jcb[1, 0] = np.dot(dNdt_V_n[i, :], x_V[icon_V[:, iel]])
            jcb[1, 1] = np.dot(dNdt_V_n[i, :], z_V[icon_V[:, iel]])
            jcbi = np.linalg.inv(jcb)
            jcbi00n[iel, i] = jcbi[0, 0]
            jcbi01n[iel, i] = jcbi[0, 1]
            jcbi10n[iel, i] = jcbi[1, 0]
            jcbi11n[iel, i] = jcbi[1, 1]
        # end for
    # end for

    return N_P_n, dNdr_V_n, dNdt_V_n, jcbi00n, jcbi01n, jcbi10n, jcbi11n


###################################################################################################
