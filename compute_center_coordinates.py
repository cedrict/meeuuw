###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################


def compute_center_coordinates(geometry, nel, x_V, z_V, icon_V):

    x_e = np.zeros(nel, dtype=np.float64)
    z_e = np.zeros(nel, dtype=np.float64)
    rad_e = np.zeros(nel, dtype=np.float64)
    theta_e = np.zeros(nel, dtype=np.float64)

    for iel in range(0, nel):
        x_e[iel] = x_V[icon_V[8, iel]]
        z_e[iel] = z_V[icon_V[8, iel]]
        match geometry:
            case "quarter" | "half" | "eighth":
                rad_e[iel] = np.sqrt(x_e[iel] ** 2 + z_e[iel] ** 2)
                theta_e[iel] = np.pi / 2 - np.arctan2(x_e[iel], z_e[iel])
            case "annulus":
                rad_e[iel] = np.sqrt(x_e[iel] ** 2 + z_e[iel] ** 2)
                theta_e[iel] = np.arctan2(z_e[iel], x_e[iel])
    # end for

    return x_e, z_e, rad_e, theta_e


###################################################################################################
