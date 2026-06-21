###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

from basis_functions import basis_functions_T_dr, basis_functions_T_dt

###################################################################################################


def basis_functions_setup_Tnodes(m_T, nel, r_T, t_T, x_T, z_T, icon_T):

    jcb = np.zeros((2, 2), dtype=np.float64)
    dNdr_T_n = np.zeros((m_T, m_T), dtype=np.float64)
    dNdt_T_n = np.zeros((m_T, m_T), dtype=np.float64)
    jcbi00n = np.zeros((nel, m_T), dtype=np.float64)
    jcbi01n = np.zeros((nel, m_T), dtype=np.float64)
    jcbi10n = np.zeros((nel, m_T), dtype=np.float64)
    jcbi11n = np.zeros((nel, m_T), dtype=np.float64)

    for iel in range(0, nel):
        for i in range(0, m_T):
            if iel == 0:
                dNdr_T_n[i, 0:m_T] = basis_functions_T_dr(r_T[i], t_T[i])
                dNdt_T_n[i, 0:m_T] = basis_functions_T_dt(r_T[i], t_T[i])
            jcb[0, 0] = np.dot(dNdr_T_n[i, :], x_T[icon_T[:, iel]])
            jcb[0, 1] = np.dot(dNdr_T_n[i, :], z_T[icon_T[:, iel]])
            jcb[1, 0] = np.dot(dNdt_T_n[i, :], x_T[icon_T[:, iel]])
            jcb[1, 1] = np.dot(dNdt_T_n[i, :], z_T[icon_T[:, iel]])
            jcbi = np.linalg.inv(jcb)
            jcbi00n[iel, i] = jcbi[0, 0]
            jcbi01n[iel, i] = jcbi[0, 1]
            jcbi10n[iel, i] = jcbi[1, 0]
            jcbi11n[iel, i] = jcbi[1, 1]
        # end for
    # end for

    return dNdr_T_n, dNdt_T_n, jcbi00n, jcbi01n, jcbi10n, jcbi11n


###################################################################################################
