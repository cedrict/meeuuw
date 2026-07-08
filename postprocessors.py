###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numba
import numpy as np

from basis_functions import basis_functions_V

###################################################################################################


@numba.njit
def compute_global_quantities(
    nel,
    nq_per_element,
    xq,
    zq,
    uq,
    wq,
    Tq,
    rhoq,
    hcapaq,
    alphaq,
    etaq,
    exxq,
    ezzq,
    exzq,
    dxxq,
    dzzq,
    dxzq,
    volume,
    JxWq,
    gxq,
    gzq,
):
    """
    Args:
    Returns:
    """

    TM = 0  # Total mass
    EK = 0  # Kinetic Energy
    WAG = 0  # Work against gravity
    TVD = 0  # Total viscous dissipation
    GPE = 0  # Gravitational potential energy
    ITE = 0  # Internal thermal energy
    vrms = 0  # root mean square velocity
    Tavrg = 0  # average temperature
    num = 0.0  # see eq 1 of chri83
    denom = 0.0  # see eq 1 of chri83

    for iel in range(0, nel):
        for iq in range(0, nq_per_element):
            eII = exxq[iel, iq] ** 2 + ezzq[iel, iq] ** 2 + 2 * exzq[iel, iq] ** 2
            TM += rhoq[iel, iq] * JxWq[iel, iq]
            EK += 0.5 * rhoq[iel, iq] * (uq[iel, iq] ** 2 + wq[iel, iq] ** 2) * JxWq[iel, iq]
            WAG -= rhoq[iel, iq] * (uq[iel, iq] * gxq[iel, iq] + wq[iel, iq] * gzq[iel, iq]) * JxWq[iel, iq]
            TVD += (
                2
                * etaq[iel, iq]
                * (
                    2 * exxq[iel, iq] ** 2 / 3
                    + 2 * ezzq[iel, iq] ** 2 / 3
                    - 2 * exxq[iel, iq] * ezzq[iel, iq] / 3
                    + 2 * exzq[iel, iq] ** 2
                )
                * JxWq[iel, iq]
            )
            # GPE+=rhoq[iel,iq]*gzq[iel,iq]*(Lz-zq[iel,iq])                      *JxWq[iel,iq]
            ITE += rhoq[iel, iq] * hcapaq[iel, iq] * Tq[iel, iq] * JxWq[iel, iq]
            vrms += (uq[iel, iq] ** 2 + wq[iel, iq] ** 2) * JxWq[iel, iq]
            Tavrg += Tq[iel, iq] * JxWq[iel, iq]
            num += etaq[iel, iq] * eII * JxWq[iel, iq]
            denom += eII * JxWq[iel, iq]
        # end for iq
    # end for iel
    vrms = np.sqrt(vrms / volume)
    Tavrg /= volume
    eta_avrg = num / denom

    return vrms, EK, WAG, TVD, GPE, ITE, TM, Tavrg, eta_avrg

###################################################################################################
# TODO: add avrg over face

def compute_boundary_velocity_statistics(x_V,z_V,u,w,left_Vnodes,right_Vnodes,bottom_Vnodes,top_Vnodes):

    vel_left=np.sqrt(u[left_Vnodes]**2+w[left_Vnodes]**2)
    vel_min_left=np.min(vel_left)
    vel_max_left=np.max(vel_left)

    vel_right=np.sqrt(u[right_Vnodes]**2+w[right_Vnodes]**2)
    vel_min_right=np.min(vel_right)
    vel_max_right=np.max(vel_right)

    vel_bottom=np.sqrt(u[bottom_Vnodes]**2+w[bottom_Vnodes]**2)
    vel_min_bottom=np.min(vel_bottom)
    vel_max_bottom=np.max(vel_bottom)

    vel_top=np.sqrt(u[top_Vnodes]**2+w[top_Vnodes]**2)
    vel_min_top=np.min(vel_top)
    vel_max_top=np.max(vel_top)

    return vel_max_left,vel_max_right,vel_max_bottom,vel_max_top



###################################################################################################


def compute_Nu(
    Lx,
    Lz,
    nel,
    top_element,
    bottom_element,
    icon_V,
    T,
    dTdz_nodal,
    nq_per_dim,
    qcoords,
    qweights,
    hx,
):
    """
    Args:
    Returns:
    """

    avrg_T_top = 0
    avrg_dTdz_top = 0
    avrg_T_bottom = 0
    avrg_dTdz_bottom = 0

    jcob = hx / 2

    for iel in range(0, nel):
        if top_element[iel]:
            sq = +1
            ny = +1
            for iq in range(0, nq_per_dim):
                rq = qcoords[iq]
                N = basis_functions_V(rq, sq)
                Tq = np.dot(N, T[icon_V[:, iel]])
                dTdzq = np.dot(N, dTdz_nodal[icon_V[:, iel]])
                avrg_T_top += Tq * jcob * qweights[iq]
                avrg_dTdz_top += dTdzq * jcob * qweights[iq] * ny
            # end for
        # end if

        if bottom_element[iel]:
            sq = -1
            ny = -1
            for iq in range(0, nq_per_dim):
                rq = qcoords[iq]
                N = basis_functions_V(rq, sq)
                Tq = np.dot(N, T[icon_V[:, iel]])
                dTdzq = np.dot(N, dTdz_nodal[icon_V[:, iel]])
                avrg_T_bottom += Tq * jcob * qweights[iq]
                avrg_dTdz_bottom += dTdzq * jcob * qweights[iq] * ny
            # end for
        # end if
    # end for

    avrg_T_top /= Lx
    avrg_T_bottom /= Lx
    avrg_dTdz_top /= Lx
    avrg_dTdz_bottom /= Lx

    Nu = np.abs(avrg_dTdz_top) / (avrg_T_bottom-avrg_T_top) * Lz

    return avrg_T_bottom, avrg_T_top, avrg_dTdz_bottom, avrg_dTdz_top, Nu


###################################################################################################
# analytical solution


def ms_velocity_x(x, y, exp):
    match exp:
        case 19:
            val = x * x * (1.0 - x) ** 2 * (2.0 * y - 6.0 * y * y + 4 * y * y * y)
        case 21:
            import solcx

            uu, vv, pp = solcx.SolCxSolution(x, y)
            val = -uu
        case 22:
            import solkz

            uu, vv, pp = solkz.SolKzSolution(x, y)
            val = -uu
    return val


def ms_velocity_z(x, y, exp):
    match exp:
        case 19:
            val = -y * y * (1.0 - y) ** 2 * (2.0 * x - 6.0 * x * x + 4 * x * x * x)
        case 21:
            import solcx

            uu, vv, pp = solcx.SolCxSolution(x, y)
            val = -vv
        case 22:
            import solkz

            uu, vv, pp = solkz.SolKzSolution(x, y)
            val = -vv
    return val


def ms_pressure(x, y, exp):
    match exp:
        case 19:
            val = x * (1.0 - x) - 1.0 / 6.0
        case 21:
            import solcx

            uu, vv, pp = solcx.SolCxSolution(x, y)
            val = -pp
        case 22:
            import solkz

            uu, vv, pp = solkz.SolKzSolution(x, y)
            val = -pp
    return val


###################################################################################################


# @numba.njit
def compute_discretisation_errors(nel, nq_per_element, xq, zq, uq, wq, pq, volume, JxWq, exp):
    """
    Args:
    Returns:
    """

    errv = 0.0
    errp = 0.0
    for iel in range(0, nel):
        for iq in range(0, nq_per_element):
            erruq2 = (uq[iel, iq] - ms_velocity_x(xq[iel, iq], zq[iel, iq], exp)) ** 2
            errwq2 = (wq[iel, iq] - ms_velocity_z(xq[iel, iq], zq[iel, iq], exp)) ** 2
            errv += (erruq2 + errwq2) * JxWq[iel, iq]
            errpq2 = (pq[iel, iq] - ms_pressure(xq[iel, iq], zq[iel, iq], exp)) ** 2
            errp += errpq2 * JxWq[iel, iq]

    errv = np.sqrt(errv)
    errp = np.sqrt(errp)

    return errv, errp


###################################################################################################
