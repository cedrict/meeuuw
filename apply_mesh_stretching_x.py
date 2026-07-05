###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################


def apply_mesh_stretching_x(m_V, x_V, x_P, nelx, nelz, icon_V, icon_P, x_segments, nelx_segments, Lx):
    """
    Args:
        a:
    Returns:
        bla
    """

    ######################################################
    # we start by building the xi array, i.e. the
    # dimensionless horizontal coordinates in [0,1]

    match m_V:
        case 5:
            nnx = nelx + 1
            nnz = nelz + 1
            xi = np.zeros(nnx)
            c = 0
            iseg = 0
            for ielx in range(0, nelx):
                h = (x_segments[iseg + 1] - x_segments[iseg]) / nelx_segments[iseg]
                xi[ielx + 1] = xi[ielx] + h
                c += 1
                if c == nelx_segments[iseg]:
                    c = 0
                    iseg += 1
        case 9:
            nnx = 2 * nelx + 1
            nnz = 2 * nelz + 1
            xi = np.zeros(nnx)
            c = 0
            iseg = 0
            for ielx in range(0, nelx):
                h = (x_segments[iseg + 1] - x_segments[iseg]) / nelx_segments[iseg]
                xi[2 * (ielx + 1)] = xi[2 * ielx] + h
                xi[2 * ielx + 1] = xi[2 * (ielx + 1)] - h / 2
                c += 1
                if c == nelx_segments[iseg]:
                    c = 0
                    iseg += 1

    # np.savetxt("DEBUG/xi.ascii", np.array([xi]).T)

    ######################################################
    # stretch V mesh

    for k in range(0, nnz):
        for i in range(0, nnx):
            inode = k * nnx + i
            x_V[inode] = xi[i] * Lx

    ###############################################
    # now that the new velocity mesh is done, we
    # need to adapt the pressure mesh (Q1 space).

    x_P[icon_P[0, :]] = x_V[icon_V[0, :]]
    x_P[icon_P[1, :]] = x_V[icon_V[1, :]]
    x_P[icon_P[2, :]] = x_V[icon_V[2, :]]
    x_P[icon_P[3, :]] = x_V[icon_V[3, :]]

    ###############################################
    # same for temperature
    # if Q1+xQ1 elements are used the T space is Q1 (same as pressure)
    # if Q2xQ1 elements are used the T space is Q2 (same as velocity)

    match m_V:
        case 5:
            x_T = np.copy(x_P)
        case 9:
            x_T = np.copy(x_V)

    return x_V, x_P, x_T, xi


###################################################################################################
