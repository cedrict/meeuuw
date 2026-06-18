###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################


def build_pressure_connectivity(geometry, nelx, nelz, nel, m_P):

    icon_P = np.zeros((m_P, nel), dtype=np.int32)

    match geometry:
        case "box" | "eighth" | "quarter" | "half":
            counter = 0
            for j in range(0, nelz):
                for i in range(0, nelx):
                    icon_P[0, counter] = i + j * (nelx + 1)
                    icon_P[1, counter] = i + 1 + j * (nelx + 1)
                    icon_P[2, counter] = i + 1 + (j + 1) * (nelx + 1)
                    icon_P[3, counter] = i + (j + 1) * (nelx + 1)
                    counter += 1
                # end for
            # end for
        case "annulus":
            nelt = nelx
            nelr = nelz
            counter = 0
            for j in range(0, nelr):
                for i in range(0, nelt):
                    icon1 = counter
                    icon2 = counter + 1
                    icon3 = i + (j + 1) * nelt + 1
                    icon4 = i + (j + 1) * nelt
                    if i == nelt - 1:
                        icon2 -= nelt
                        icon3 -= nelt
                    icon_P[0, counter] = icon2
                    icon_P[1, counter] = icon1
                    icon_P[2, counter] = icon4
                    icon_P[3, counter] = icon3
                    counter += 1
                # end for
        case _:
            raise ValueError("build_pressure_mesh: unknown geometry")

    return icon_P


###################################################################################################
