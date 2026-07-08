###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################


def prescribe_initial_topography(
    experiment,
    Lx,Lz,
    nn_V,
    nelx,
    nelz,
    x_V,
    z_V,
    z_P,
    z_T,
    icon_V,
    icon_P,
    top_Vnodes,
    m_V,
    use_stretching,
):
    """
    Args:
        a:
        b:
    Returns:
        bla
    """

    # store temporarily current z_V

    z_V_old = np.copy(z_V)

    ###############################################

    match experiment:
        case 6:
            for i in range(nn_V):
                if top_Vnodes[i]:
                    z_V[i] = 700e3 + 7e3 * np.cos(x_V[i] / Lx * np.pi)
        case _:
            return

    ###############################################
    # resample per column

    match m_V:
        case 5:
            nnx = nelx + 1
            nnz = nelz + 1
        case 9:
            nnx = 2 * nelx + 1
            nnz = 2 * nelz + 1

    if use_stretching:

       1 ####################### XXXXXXXXXXXXXXXXXXXXXXX

    else:

       for k in range(1, nnz - 1):  # I do not need botom & top rows
           for i in range(0, nnx):
               inode = i + nnx * k
               inode_top = i + nnx * (nnz - 1)
               ratio = z_V[inode] / z_V_old[inode_top]
               z_V[inode] = ratio * z_V[inode_top]

    ###############################################
    # In the case of Q1+ velocity space, make sure
    # that the bubble node is in the middle of the
    # element. No need to tamper with x coords.
    # reminder of the local numbering of nodes
    # 3-----2
    # |  4  |
    # 0-----1
    # In the case of Q2 velocity space make sure
    # mid-edge nodes 4,5,6,7 and center node 9 are
    # in the middle. No need to tamper with x coords.
    # reminder of the local numbering of nodes
    # 3--6--2
    # 7  8  5
    # 0--4--1

    nel = nelx * nelz

    match m_V:
        case 5:
            z_V[icon_V[4, :]] = 0.25 * (z_V[icon_V[0, :]] + z_V[icon_V[1, :]] + z_V[icon_V[2, :]] + z_V[icon_V[3, :]])

        case 9:
            z_V[icon_V[4, :]] = 0.5 * (z_V[icon_V[0, :]] + z_V[icon_V[1, :]])  # bottom edge
            z_V[icon_V[5, :]] = 0.5 * (z_V[icon_V[1, :]] + z_V[icon_V[2, :]])  # right edge
            z_V[icon_V[6, :]] = 0.5 * (z_V[icon_V[2, :]] + z_V[icon_V[3, :]])  # top edge
            z_V[icon_V[7, :]] = 0.5 * (z_V[icon_V[3, :]] + z_V[icon_V[0, :]])  # left edge
            z_V[icon_V[8, :]] = 0.5 * (z_V[icon_V[4, :]] + z_V[icon_V[6, :]])  # center

    ###############################################
    # now that the new velocity mesh is done, we
    # need to adapt the pressure mesh (Q1 space).

    z_P[icon_P[0, :]] = z_V[icon_V[0, :]]
    z_P[icon_P[1, :]] = z_V[icon_V[1, :]]
    z_P[icon_P[2, :]] = z_V[icon_V[2, :]]
    z_P[icon_P[3, :]] = z_V[icon_V[3, :]]

    ###############################################
    # same for temperature

    match m_V:
        case 5:
            z_T = np.copy(z_P)
        case 9:
            z_T = np.copy(z_V)

    return z_V, z_P, z_T


###################################################################################################
