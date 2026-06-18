###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################


def build_pressure_mesh(geometry, nn_P, nelx, nelz, hx, hz, Rinner, Router, opening_angle, debug):

    x_P = np.zeros(nn_P, dtype=np.float64)
    z_P = np.zeros(nn_P, dtype=np.float64)
    rad_P = np.zeros(nn_P, dtype=np.float64)
    theta_P = np.zeros(nn_P, dtype=np.float64)
    top_Pnodes = np.zeros(nn_P, dtype=bool)
    bot_Pnodes = np.zeros(nn_P, dtype=bool)

    match geometry:
        case "box" | "eighth" | "quarter" | "half":
            counter = 0
            for j in range(0, nelz + 1):
                for i in range(0, nelx + 1):
                    x_P[counter] = i * hx
                    z_P[counter] = j * hz
                    if j == 0:
                        bot_Pnodes[counter] = True
                    if j == nelz:
                        top_Pnodes[counter] = True
                    counter += 1
                # end for
            # end for
        case "annulus":
            counter = 0
            for j in range(0, nelz + 1):
                for i in range(0, nelx):
                    x_P[counter] = i * hx
                    z_P[counter] = j * hz
                    if j == 0:
                        bot_Pnodes[counter] = True
                    if j == nelz:
                        top_Pnodes[counter] = True
                    counter += 1
                # end for
            # end for
        case _:
            raise ValueError("build_pressure_mesh: unknown geometry")

    match geometry:
        case "quarter" | "half" | "eighth":
            for i in range(0, nn_P):
                rad_P[i] = Rinner + z_P[i] * (Router - Rinner)
                theta_P[i] = np.pi / 2 - x_P[i] * opening_angle
                x_P[i] = rad_P[i] * np.cos(theta_P[i])
                z_P[i] = rad_P[i] * np.sin(theta_P[i])
        case "annulus":
            for i in range(0, nn_P):
                xi = x_P[i]
                zi = z_P[i]
                t = xi * 2 * np.pi
                x_P[i] = np.cos(t) * (Rinner + zi * (Router - Rinner))
                z_P[i] = np.sin(t) * (Rinner + zi * (Router - Rinner))
                rad_P[i] = Rinner + zi * (Router - Rinner)
                theta_P[i] = np.arctan2(z_P[i], x_P[i])
                if theta_P[i] < 0.0:
                    theta_P[i] += 2.0 * np.pi

    if debug:
        np.savetxt("DEBUG/mesh_P.ascii", np.array([x_P, z_P]).T, header="# x,z")

    return x_P, z_P, rad_P, theta_P, top_Pnodes, bot_Pnodes


###################################################################################################
