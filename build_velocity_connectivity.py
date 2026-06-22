###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################
# TODO: annulus geometry

def build_velocity_connectivity(geometry, m_V, nelx, nelz, nnx, nnz, middleH_nodes, middleV_nodes):

    nel = nelx * nelz

    icon_V = np.zeros((m_V, nel), dtype=np.int32)
    top_element = np.zeros(nel, dtype=bool)
    bot_element = np.zeros(nel, dtype=bool)
    left_element = np.zeros(nel, dtype=bool)
    right_element = np.zeros(nel, dtype=bool)
    middleH_element = np.zeros(nel, dtype=bool)
    middleV_element = np.zeros(nel, dtype=bool)

    match m_V:
        case 5:  # velocity FE space is Q1+
            match geometry:
                case "box" | "eighth" | "quarter" | "half":
                    counter = 0
                    for j in range(0, nelz):
                        for i in range(0, nelx):
                            icon_V[0, counter] = i + j * (nelx + 1)
                            icon_V[1, counter] = i + 1 + j * (nelx + 1)
                            icon_V[2, counter] = i + 1 + (j + 1) * (nelx + 1)
                            icon_V[3, counter] = i + (j + 1) * (nelx + 1)
                            icon_V[4, counter] = (nelx + 1) * (nelz + 1) + counter
                            if i == 0:
                                left_element[counter] = True
                            if i == nelx - 1:
                                right_element[counter] = True
                            if j == 0:
                                bot_element[counter] = True
                            if j == nelz - 1:
                                top_element[counter] = True
                            if middleH_nodes[icon_V[0, counter]]:
                                middleH_element[counter] = True
                            if middleH_nodes[icon_V[2, counter]]:
                                middleH_element[counter] = True
                            if middleV_nodes[icon_V[0, counter]]:
                                middleV_element[counter] = True
                            if middleV_nodes[icon_V[1, counter]]:
                                middleV_element[counter] = True
                            counter += 1

                # case "annulus":

                case _:
                    raise ValueError("build_velocity_connectivity: unknown geometry")

        case 9:  # velocity FE space is Q2
            match geometry:
                case "box" | "eighth" | "quarter" | "half":
                    counter = 0
                    for j in range(0, nelz):
                        for i in range(0, nelx):
                            icon_V[0, counter] = i * 2 + 1 + j * 2 * nnx - 1
                            icon_V[1, counter] = i * 2 + 3 + j * 2 * nnx - 1
                            icon_V[2, counter] = i * 2 + 3 + j * 2 * nnx + nnx * 2 - 1
                            icon_V[3, counter] = i * 2 + 1 + j * 2 * nnx + nnx * 2 - 1
                            icon_V[4, counter] = i * 2 + 2 + j * 2 * nnx - 1
                            icon_V[5, counter] = i * 2 + 3 + j * 2 * nnx + nnx - 1
                            icon_V[6, counter] = i * 2 + 2 + j * 2 * nnx + nnx * 2 - 1
                            icon_V[7, counter] = i * 2 + 1 + j * 2 * nnx + nnx - 1
                            icon_V[8, counter] = i * 2 + 2 + j * 2 * nnx + nnx - 1
                            if i == 0:
                                left_element[counter] = True
                            if i == nelx - 1:
                                right_element[counter] = True
                            if j == 0:
                                bot_element[counter] = True
                            if j == nelz - 1:
                                top_element[counter] = True
                            if middleH_nodes[icon_V[0, counter]]:
                                middleH_element[counter] = True
                            if middleH_nodes[icon_V[2, counter]]:
                                middleH_element[counter] = True
                            if middleV_nodes[icon_V[0, counter]]:
                                middleV_element[counter] = True
                            if middleV_nodes[icon_V[1, counter]]:
                                middleV_element[counter] = True
                            counter += 1
                        # end for
                    # end for
                case "annulus":
                    nelt = nelx
                    nelr = nelz
                    counter = 0
                    for j in range(0, nelr):
                        for i in range(0, nelt):
                            icon_V[0, counter] = 2 * counter + 2 + 2 * j * nelt
                            icon_V[1, counter] = 2 * counter + 2 * j * nelt
                            icon_V[2, counter] = icon_V[1, counter] + 4 * nelt
                            icon_V[3, counter] = icon_V[1, counter] + 4 * nelt + 2
                            icon_V[4, counter] = icon_V[0, counter] - 1
                            icon_V[5, counter] = icon_V[1, counter] + 2 * nelt
                            icon_V[6, counter] = icon_V[2, counter] + 1
                            icon_V[7, counter] = icon_V[5, counter] + 2
                            icon_V[8, counter] = icon_V[5, counter] + 1
                            if i == nelt - 1:
                                icon_V[0, counter] -= 2 * nelt
                                icon_V[7, counter] -= 2 * nelt
                                icon_V[3, counter] -= 2 * nelt
                            if j == 0:
                                bot_element[counter] = True
                            if j == nelr - 1:
                                top_element[counter] = True
                            counter += 1
                        # end for
                    # end for

                case _:
                    raise ValueError("build_velocity_connectivity: unknown geometry")

        case _:
            raise ValueError("build_velocity_mesh: unknown m_V value")

    return (
        icon_V,
        top_element,
        bot_element,
        left_element,
        right_element,
        middleH_element,
        middleV_element,
    )


###################################################################################################
