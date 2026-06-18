###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numba
import numpy as np

## PB: shear heating should use dev strain rate

###################################################################################################


@numba.njit
def build_matrix_energy(
    bignb,
    nel,
    nq_per_element,
    m_T,
    Nfem_T,
    T,
    icon_V,
    rhoq,
    etaq,
    Tq,
    uq,
    wq,
    hcondq,
    hcapaq,
    alphaq,
    hprodq,
    exxq,
    ezzq,
    exzq,
    dpdxq,
    dpdzq,
    JxWq,
    N_V,
    dNdr_V,
    dNds_V,
    jcbi00q,
    jcbi01q,
    jcbi10q,
    jcbi11q,
    bc_fix_T,
    bc_val_T,
    dt,
    formulation,
    rho0,
):

    VV_T = np.zeros(bignb, dtype=np.float64)
    Tvect = np.zeros(m_T, dtype=np.float64)
    rhs = np.zeros(Nfem_T, dtype=np.float64)
    B = np.zeros((2, m_T), dtype=np.float64)

    counter = 0
    for iel in range(0, nel):
        b_el = np.zeros(m_T, dtype=np.float64)  # elemental rhs
        A_el = np.zeros((m_T, m_T), dtype=np.float64)  # elemental matrix
        Ka = np.zeros((m_T, m_T), dtype=np.float64)  # elemental advection matrix
        Kd = np.zeros((m_T, m_T), dtype=np.float64)  # elemental diffusion matrix
        MM = np.zeros((m_T, m_T), dtype=np.float64)  # elemental mass matrix
        velq = np.zeros((1, 2), dtype=np.float64)  # velocity at quad point

        Tvect[0:m_T] = T[icon_V[0:m_T, iel]]

        for iq in range(0, nq_per_element):
            N = N_V[iq, :]

            velq[0, 0] = uq[iel, iq]
            velq[0, 1] = wq[iel, iq]

            dNdx = jcbi00q[iel, iq] * dNdr_V[iq, :] + jcbi01q[iel, iq] * dNds_V[iq, :]
            dNdy = jcbi10q[iel, iq] * dNdr_V[iq, :] + jcbi11q[iel, iq] * dNds_V[iq, :]

            B[0, :] = dNdx
            B[1, :] = dNdy

            MM += np.outer(N, N) * rho0 * hcapaq[iel, iq] * JxWq[iel, iq]  # mass matrix

            Kd += B.T.dot(B) * hcondq[iel, iq] * JxWq[iel, iq]  # diffusion matrix

            Ka += np.outer(N, velq.dot(B)) * rho0 * hcapaq[iel, iq] * JxWq[iel, iq]  # advection matrix

            b_el[:] += N[:] * hprodq[iel, iq] * JxWq[iel, iq]

            if formulation == "EBA":
                # viscous dissipation
                b_el[:] += (
                    N[:]
                    * 2
                    * etaq[iel, iq]
                    * (
                        2.0 / 3.0 * exxq[iel, iq] ** 2
                        + 2.0 / 3.0 * ezzq[iel, iq] ** 2
                        - 2.0 / 3.0 * exxq[iel, iq] * ezzq[iel, iq]
                        + 2 * exzq[iel, iq] ** 2
                    )
                    * JxWq[iel, iq]
                )
                # adiabatic heating
                b_el[:] += (
                    N[:]
                    * alphaq[iel, iq]
                    * Tq[iel, iq]
                    * (velq[0, 0] * dpdxq[iel, iq] + velq[0, 1] * dpdzq[iel, iq])
                    * JxWq[iel, iq]
                )
        # end for
        b_el *= dt

        A_el += MM + (Ka + Kd) * dt * 0.5
        b_el += (MM - (Ka + Kd) * dt * 0.5).dot(Tvect)

        # apply boundary conditions
        for k1 in range(0, m_T):
            m1 = icon_V[k1, iel]
            if bc_fix_T[m1]:
                Aref = A_el[k1, k1]
                for k2 in range(0, m_T):
                    b_el[k2] -= A_el[k2, k1] * bc_val_T[m1]
                    A_el[k1, k2] = 0
                    A_el[k2, k1] = 0
                # end for
                A_el[k1, k1] = Aref
                b_el[k1] = Aref * bc_val_T[m1]
            # end for
        # end for

        # assemble matrix K_mat and right hand side rhs
        for ikk in range(m_T):
            m1 = icon_V[ikk, iel]
            for jkk in range(m_T):
                VV_T[counter] = A_el[ikk, jkk]
                counter += 1
            rhs[m1] += b_el[ikk]
        # end for

    return VV_T, rhs


###################################################################################################
