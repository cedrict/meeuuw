###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numba

from project_nodal_field_onto_qpoints import project_nodal_Vfield_onto_qpoints

###################################################################################################


@numba.njit
def remove_net_rotation(nq_per_element, nel, icon_V, xq, zq, u, w, nn_V, N_V, x_V, z_V, JxWq):

    uq = project_nodal_Vfield_onto_qpoints(u, nq_per_element, nel, m_V, N_V, icon_V)
    wq = project_nodal_Vfield_onto_qpoints(w, nq_per_element, nel, m_V, N_V, icon_V)

    Izz = 0.0
    Hz = 0.0
    for iel in range(0, nel):
        for iq in range(0, nq_per_element):
            Hz += (xq[iel, iq] * wq[iel, iq] - zq[iel, iq] * uq[iel, iq]) * JxWq[iel, iq]
            Izz += (xq[iel, iq] ** 2 + zq[iel, iq] ** 2) * JxWq[iel, iq]
    # end for iel
    omega_z = Hz / Izz

    for i in range(0, nn_V):
        u[i] -= -z_V[i] * omega_z
        w[i] -= +x_V[i] * omega_z

    return u, w, omega_z


###################################################################################################
