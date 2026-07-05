###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################


def apply_mesh_stretching_z(m_V, z_V, z_P, nelx, nelz, icon_V, icon_P, z_segments, nelz_segments, Lz):
    """
    Args:
        a:
    Returns:
        bla
    """

    ######################################################
    # we start by building the zeta array, i.e. the
    # dimensionless vertical coordinates in [0,1]

    match m_V:
        case 5:
            nnx = nelx + 1
            nnz = nelz + 1
            zeta = np.zeros(nnz)
            c = 0
            iseg = 0
            for ielz in range(0, nelz):
                h = (z_segments[iseg + 1] - z_segments[iseg]) / nelz_segments[iseg]
                zeta[ielz + 1] = zeta[ielz] + h
                c += 1
                if c == nelz_segments[iseg]:
                    c = 0
                    iseg += 1
        case 9:
            nnx = 2 * nelx + 1
            nnz = 2 * nelz + 1
            zeta = np.zeros(nnz)
            c = 0
            iseg = 0
            for ielz in range(0, nelz):
                h = (z_segments[iseg + 1] - z_segments[iseg]) / nelz_segments[iseg]
                zeta[2 * (ielz + 1)] = zeta[2 * ielz] + h
                zeta[2 * ielz + 1] = zeta[2 * (ielz + 1)] - h / 2
                c += 1
                if c == nelz_segments[iseg]:
                    c = 0
                    iseg += 1

    #np.savetxt("DEBUG/zeta.ascii", np.array([zeta]).T)

    ######################################################
    # stretch V mesh

    for k in range(0, nnz):
        for i in range(0, nnx):
            inode = k * nnx + i
            z_V[inode] = zeta[k] * Lz

    ###############################################
    # now that the new velocity mesh is done, we
    # need to adapt the pressure mesh (Q1 space).

    z_P[icon_P[0, :]] = z_V[icon_V[0, :]]
    z_P[icon_P[1, :]] = z_V[icon_V[1, :]]
    z_P[icon_P[2, :]] = z_V[icon_V[2, :]]
    z_P[icon_P[3, :]] = z_V[icon_V[3, :]]

    ###############################################
    # same for temperature
    # if Q1+xQ1 elements are used the T space is Q1 (same as pressure)
    # if Q2xQ1 elements are used the T space is Q2 (same as velocity)

    match m_V:
        case 5:
            z_T = np.copy(z_P)
        case 9:
            z_T = np.copy(z_V)

    return z_V, z_P, z_T, zeta


###################################################################################################
