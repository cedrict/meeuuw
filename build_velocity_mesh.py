###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

from constants import eps

###################################################################################################
# BL: bottom left, BR: bottom right, TL: top left, TR: top right
# if geometry is 'eighth', 'quarter' or 'half' we still need to set Lx=Lz=1
# Note that in the case of the annulus there are no left or right boundaries.
###################################################################################################
# TODO: annulus geometry


def build_velocity_mesh(geometry, m_V, nn_V, nelx, nelz, Lx, Lz, Rinner, Router, opening_angle, debug_ascii):

    hx = Lx / nelx
    hz = Lz / nelz

    #######################################################

    match m_V:
        case 5:  # velocity FE space is Q1+
            match geometry:
                case "box" | "eighth" | "quarter" | "half":
                    nnx = nelx + 1
                    nnz = nelz + 1
                case "annulus":
                    nnx = nelx
                    nnz = nelz + 1
                case _:
                    raise ValueError("build_velocity_mesh: unknown geometry for m_V=5")
        case 9:  # velocity FE space is Q2
            match geometry:
                case "box" | "eighth" | "quarter" | "half":
                    nnx = 2 * nelx + 1
                    nnz = 2 * nelz + 1
                case "annulus":
                    nnx = 2 * nelx
                    nnz = 2 * nelz + 1
                case _:
                    raise ValueError("build_velocity_mesh: unknown geometry for m_V=9")
        case _:
            raise ValueError("build_velocity_mesh: unknown m_V value")

    #######################################################
    x_V = np.zeros(nn_V, dtype=np.float64)
    z_V = np.zeros(nn_V, dtype=np.float64)
    top_Vnodes = np.zeros(nn_V, dtype=bool)
    bot_Vnodes = np.zeros(nn_V, dtype=bool)
    left_Vnodes = np.zeros(nn_V, dtype=bool)
    right_Vnodes = np.zeros(nn_V, dtype=bool)
    middleH_nodes = np.zeros(nn_V, dtype=bool)
    middleV_nodes = np.zeros(nn_V, dtype=bool)
    hull_Vnodes = np.zeros(nn_V, dtype=bool)

    match m_V:
        case 5:  # velocity FE space is Q1+
            match geometry:
                case "box" | "eighth" | "quarter" | "half":
                    counter = 0
                    for j in range(0, nelz + 1):
                        for i in range(0, nelx + 1):
                            x_V[counter] = i * hx
                            z_V[counter] = j * hz
                            if i == 0:
                                left_Vnodes[counter] = True
                            if i == 2 * nelx:
                                right_Vnodes[counter] = True
                            if j == 0:
                                bot_Vnodes[counter] = True
                            if j == 2 * nelz:
                                top_Vnodes[counter] = True
                            if (
                                top_Vnodes[counter]
                                or bot_Vnodes[counter]
                                or right_Vnodes[counter]
                                or left_Vnodes[counter]
                            ):
                                hull_Vnodes[counter] = True
                            if abs(x_V[counter] / Lx - 0.5) < eps:
                                middleV_nodes[counter] = True
                            if abs(z_V[counter] / Lz - 0.5) < eps:
                                middleH_nodes[counter] = True
                            if i == 0 and j == 0:
                                cornerBL = counter
                            if i == nnx - 1 and j == 0:
                                cornerBR = counter
                            if i == 0 and j == nnz - 1:
                                cornerTL = counter
                            if i == nnx - 1 and j == nnz - 1:
                                cornerTR = counter
                            counter += 1
                        # end for
                    # end for

                    for j in range(0, nely):
                        for i in range(0, nelx):
                            x_V[counter] = i * hx + 1 / 2.0 * hx
                            z_V[counter] = j * hy + 1 / 2.0 * hy
                            counter += 1

                # case "annulus":

                case _:
                    exit("build_velocity_mesh: unknown geometry")

        case 9:  # velocity FE space is Q2
            match geometry:
                case "box" | "eighth" | "quarter" | "half":
                    counter = 0
                    for j in range(0, 2 * nelz + 1):
                        for i in range(0, 2 * nelx + 1):
                            x_V[counter] = i * hx / 2
                            z_V[counter] = j * hz / 2
                            if i == 0:
                                left_Vnodes[counter] = True
                            if i == 2 * nelx:
                                right_Vnodes[counter] = True
                            if j == 0:
                                bot_Vnodes[counter] = True
                            if j == 2 * nelz:
                                top_Vnodes[counter] = True
                            if (
                                top_Vnodes[counter]
                                or bot_Vnodes[counter]
                                or right_Vnodes[counter]
                                or left_Vnodes[counter]
                            ):
                                hull_Vnodes[counter] = True
                            if abs(x_V[counter] / Lx - 0.5) < eps:
                                middleV_nodes[counter] = True
                            if abs(z_V[counter] / Lz - 0.5) < eps:
                                middleH_nodes[counter] = True
                            if i == 0 and j == 0:
                                cornerBL = counter
                            if i == nnx - 1 and j == 0:
                                cornerBR = counter
                            if i == 0 and j == nnz - 1:
                                cornerTL = counter
                            if i == nnx - 1 and j == nnz - 1:
                                cornerTR = counter
                            counter += 1
                        # end for
                    # end for
                case "annulus":
                    counter = 0
                    for j in range(0, 2 * nelz + 1):
                        for i in range(0, 2 * nelx):
                            x_V[counter] = i * hx / 2
                            z_V[counter] = j * hz / 2
                            if j == 0:
                                bot_Vnodes[counter] = True
                            if j == 2 * nelz:
                                top_Vnodes[counter] = True
                            if top_Vnodes[counter] or bot_Vnodes[counter]:
                                hull_Vnodes[counter] = True
                            counter += 1
                        # end for
                    # end for
                    cornerBL = 0
                    cornerBR = 0
                    cornerTL = 0
                    cornerTR = 0
                case _:
                    exit("build_velocity_mesh: unknown geometry")

        case _:
            raise ValueError("build_velocity_mesh: unknown m_V value")

    #######################################################
    # now that I have computed the cartesian coordinates of
    # the nodes, I can compute their polar coordinates.

    rad_V = np.zeros(nn_V, dtype=np.float64)
    theta_V = np.zeros(nn_V, dtype=np.float64)

    match geometry:
        case "eighth" | "quarter" | "half":
            for i in range(0, nn_V):
                rad_V[i] = Rinner + z_V[i] * (Router - Rinner)
                theta_V[i] = np.pi / 2 - x_V[i] * opening_angle
                x_V[i] = rad_V[i] * np.cos(theta_V[i])
                z_V[i] = rad_V[i] * np.sin(theta_V[i])
        case "annulus":
            for i in range(0, nn_V):
                xi = x_V[i]
                zi = z_V[i]
                t = xi * 2 * np.pi
                x_V[i] = np.cos(t) * (Rinner + zi * (Router - Rinner))
                z_V[i] = np.sin(t) * (Rinner + zi * (Router - Rinner))
                rad_V[i] = Rinner + zi * (Router - Rinner)
                theta_V[i] = np.arctan2(z_V[i], x_V[i])
                if theta_V[i] < 0.0:
                    theta_V[i] += 2.0 * np.pi

    #######################################################

    if debug_ascii:
        np.savetxt("DEBUG/mesh_V.ascii", np.array([x_V, z_V]).T, header="# x,z")

    return (
        hx,
        hz,
        nnx,
        nnz,
        x_V,
        z_V,
        rad_V,
        theta_V,
        top_Vnodes,
        bot_Vnodes,
        left_Vnodes,
        right_Vnodes,
        middleH_nodes,
        middleV_nodes,
        hull_Vnodes,
        cornerBL,
        cornerBR,
        cornerTL,
        cornerTR,
    )


###################################################################################################
