###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numba
import numpy as np

###################################################################################################


@numba.njit
def compute_nodal_heat_flux(
    icon_T, T, hcond_nodal, nn_T, m_T, nel, dNdr_T_n, dNds_T_n, jcbi00_T, jcbi01_T, jcbi10_T, jcbi11_T
):

    qx_n = np.zeros(nn_T, dtype=np.float64)
    qy_n = np.zeros(nn_T, dtype=np.float64)
    dTdx_n = np.zeros(nn_T, dtype=np.float64)
    dTdy_n = np.zeros(nn_T, dtype=np.float64)
    count = np.zeros(nn_T, dtype=np.float64)

    for iel in range(0, nel):
        for i in range(0, m_T):
            inode = icon_T[i, iel]
            dNdx = jcbi00_T[iel, i] * dNdr_T_n[i, :] + jcbi01_T[iel, i] * dNds_T_n[i, :]
            dNdy = jcbi10_T[iel, i] * dNdr_T_n[i, :] + jcbi11_T[iel, i] * dNds_T_n[i, :]
            dTdx_n[inode] -= np.dot(dNdx, T[icon_T[:, iel]])
            dTdy_n[inode] -= np.dot(dNdy, T[icon_T[:, iel]])
            qx_n[inode] -= hcond_nodal[inode] * np.dot(dNdx, T[icon_T[:, iel]])
            qy_n[inode] -= hcond_nodal[inode] * np.dot(dNdy, T[icon_T[:, iel]])
            count[inode] += 1
        # end for
    # end for

    qx_n /= count
    qy_n /= count
    dTdx_n /= count
    dTdy_n /= count

    return dTdx_n, dTdy_n, qx_n, qy_n


###################################################################################################
