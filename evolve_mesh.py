###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################


def evolve_mesh_box(
    nelx,
    nelz,
    u,
    w,
    x_V,
    z_V,
    z_P,
    z_T,
    icon_V,
    icon_P,
    top_Vnodes,
    m_V,
    N_V,
    dNdr_V,
    dNdt_V,
    nq_per_dim,
    weightq,
    area,
):
    """
    The underlying assumption is that the mesh only deforms vertically: the x position
    of nodes does not change.
    Second important assumption: elements remain trapezes with parallel vertical edges.
    steps:
    1 move top row of nodes. I will start with only vertical advection - easier- but will
      later implement an advection+resampling approach.
    2 apply surface processes or small diffusion
    3 move interior nodes. I need to be careful about moving T nodes/T field!
    4 regenerate q pts and jacobians and ... ?

    Args:
        a:
        b:

    Returns:
        bla
    """

    # store temporarily current z_V

    z_V_old = np.copy(z_V)

    ###############################################
    # step 1: evolve free surface (top V nodes only)
    # at the moment only vertical movement is allowed
    # TODO: This will be revisited later on.

    z_V[top_Vnodes] += w[top_Vnodes] * dt

    ###############################################
    # step 2: surface processes

    # TODO: some day

    ###############################################
    # step 3: resample per column
    # only Q2 velocity!!

    match m_V:
        case 5:
            nnx = nelx + 1
            nnz = nelz + 1
        case 9:
            nnx = 2 * nelx + 1
            nnz = 2 * nelz + 1

    for k in range(0, nnz - 1):  # I do not need the top row
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
    # step 4: now that elements are trapezoidal,
    # we need to recompute q pts coordinates

    nq_per_element = nq_per_dim**2

    xq = project_nodal_Vfield_onto_qpoints(x_V, nq_per_element, nel, m_V, N_V, icon_V)
    zq = project_nodal_Vfield_onto_qpoints(z_V, nq_per_element, nel, m_V, N_V, icon_V)

    ###############################################
    # step 4: we now recompute the jacobian entries

    jcbi00q = np.zeros((nel, nq_per_element), dtype=np.float64)
    jcbi01q = np.zeros((nel, nq_per_element), dtype=np.float64)
    jcbi10q = np.zeros((nel, nq_per_element), dtype=np.float64)
    jcbi11q = np.zeros((nel, nq_per_element), dtype=np.float64)
    JxWq = np.zeros((nel, nq_per_element), dtype=np.float64)
    jcb = np.zeros((2, 2), dtype=np.float64)

    for iel in range(0, nel):
        cq = 0
        for iq in range(0, nq_per_dim):
            for jq in range(0, nq_per_dim):
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

    return xq, zq, JxWq, jcbi00q, jcbi01q, jcbi10q, jcbi11q, area


###################################################################################################
