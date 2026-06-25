###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np
import random
from basis_functions import *

###################################################################################################


def swarm_coordinates_setup(
    geometry,
    particle_distribution,
    nparticle,
    nparticle_per_element,
    nparticle_per_dim,
    nel,
    nq_per_dim,
    nelx,
    nelz,
    Lx,
    Lz,
    hx,
    hz,
    xq,
    zq,
    qcoords,
    x_V,
    z_V,
    icon_V,
):

    swarm_x = np.zeros(nparticle, dtype=np.float64)
    swarm_z = np.zeros(nparticle, dtype=np.float64)
    swarm_r = np.zeros(nparticle, dtype=np.float64)
    swarm_t = np.zeros(nparticle, dtype=np.float64)
    swarm_id = np.zeros(nparticle, dtype=np.int32)
    swarm_iel = np.zeros(nparticle, dtype=np.int32)
    swarm_paint = np.zeros(nparticle, dtype=np.int32)
    swarm_active = np.zeros(nparticle, dtype=bool)
    swarm_active[:] = True

    #######################################################

    match particle_distribution:
        # ----------------------------------
        case -1:  # collocated with qpoints
            counter = 0
            for iel in range(0, nel):
                c = 0
                for iq in range(0, nq_per_dim):
                    for jq in range(0, nq_per_dim):
                        swarm_x[counter] = xq[iel, c]
                        swarm_z[counter] = zq[iel, c]
                        swarm_r[counter] = qcoords[iq]
                        swarm_t[counter] = qcoords[jq]
                        swarm_iel[counter] = iel
                        swarm_id[counter] = counter
                        counter += 1
                        c += 1
                    # end for
                # end for
            # end for

        # ----------------------------------
        case 0:  # random
            counter = 0
            for iel in range(0, nel):
                for im in range(0, nparticle_per_element):
                    r = random.uniform(-1.0, +1)
                    t = random.uniform(-1.0, +1)
                    N = basis_functions_V(r, t)
                    swarm_x[counter] = np.dot(N[:], x_V[icon_V[:, iel]])
                    swarm_z[counter] = np.dot(N[:], z_V[icon_V[:, iel]])
                    swarm_r[counter] = r
                    swarm_t[counter] = t
                    swarm_iel[counter] = iel
                    swarm_id[counter] = counter
                    counter += 1
                # end for
            # end for

        # ----------------------------------
        case 1:  # regular
            counter = 0
            for iel in range(0, nel):
                for j in range(0, nparticle_per_dim):
                    for i in range(0, nparticle_per_dim):
                        r = -1.0 + i * 2.0 / nparticle_per_dim + 1.0 / nparticle_per_dim
                        t = -1.0 + j * 2.0 / nparticle_per_dim + 1.0 / nparticle_per_dim
                        N = basis_functions_V(r, t)
                        swarm_x[counter] = np.dot(N[:], x_V[icon_V[:, iel]])
                        swarm_z[counter] = np.dot(N[:], z_V[icon_V[:, iel]])
                        swarm_r[counter] = r
                        swarm_t[counter] = t
                        swarm_iel[counter] = iel
                        swarm_id[counter] = counter
                        counter += 1
                    # end for
                # end for
            # end for

        # ----------------------------------
        case 2:  # Poisson Disc
            if geometry != "box":
                exit("Poisson disc not available with this geometry")

            kpoisson = 30
            nparticle_wish = nel * nparticle_per_element  # target
            print("     -> nparticle_wish: %d " % (nparticle_wish))
            avrgdist = np.sqrt(Lx * Lz / nparticle_wish) / 1.25
            nparticle, swarm_x, swarm_z = PoissonDisc(kpoisson, avrgdist, Lx, Lz)
            print("     -> nparticle: %d " % (nparticle))

            swarm_r, swarm_t, swarm_iel = locate_particlesX(nparticle, swarm_x, swarm_z, hx, hz, x_V, z_V, icon_V, nelx)

            # swarm_id[counter] = missing!

        # ----------------------------------
        case 3:  # pseudo-random
            counter = 0
            for iel in range(0, nel):
                for j in range(0, nparticle_per_dim):
                    for i in range(0, nparticle_per_dim):
                        r = -1.0 + i * 2.0 / nparticle_per_dim + 1.0 / nparticle_per_dim
                        t = -1.0 + j * 2.0 / nparticle_per_dim + 1.0 / nparticle_per_dim
                        r += random.uniform(-0.2, +0.2) * (2 / nparticle_per_dim)
                        t += random.uniform(-0.2, +0.2) * (2 / nparticle_per_dim)
                        N = basis_functions_V(r, t)
                        swarm_x[counter] = np.dot(N[:], x_V[icon_V[:, iel]])
                        swarm_z[counter] = np.dot(N[:], z_V[icon_V[:, iel]])
                        swarm_r[counter] = r
                        swarm_t[counter] = t
                        swarm_iel[counter] = iel
                        swarm_id[counter] = counter
                        counter += 1
                    # end for
                # end for
            # end for

        # ----------------------------------
        case _:
            exit("unknown particle_distribution")

    #######################################################

    match geometry:
        case "quarter" | "half" | "eighth":
            swarm_rad = np.sqrt(swarm_x**2 + swarm_z**2)
            swarm_theta = np.pi / 2 - np.arctan2(swarm_x, swarm_z)
            print("     -> swarm_rad (m,M) %.3e %.3e " % (np.min(swarm_rad), np.max(swarm_rad)))
            print("     -> swarm_theta (m,M) %.3e %.3e " % (np.min(swarm_theta), np.max(swarm_theta)))
        case "annulus":
            swarm_rad = np.sqrt(swarm_x**2 + swarm_z**2)
            swarm_theta = np.arctan2(swarm_z, swarm_x)
        case _:
            swarm_rad = 0
            swarm_theta = 0

    #######################################################

    match geometry:
        case "box":
            for i in [0, 2, 4, 6, 8, 10, 12, 14]:
                dx = Lx / 16
                for ip in range(0, nparticle):
                    if swarm_x[ip] > i * dx and swarm_x[ip] < (i + 1) * dx:
                        swarm_paint[ip] += 1
            for i in [0, 2, 4, 6, 8, 10, 12, 14]:
                dz = Lz / 16
                for ip in range(0, nparticle):
                    if swarm_z[ip] > i * dz and swarm_z[ip] < (i + 1) * dz:
                        swarm_paint[ip] += 1

        case "quarter" | "half" | "eighth" | "annulus":
            for i in [0, 2, 4, 6, 8, 10, 12, 14]:
                drad = (Router - Rinner) / 16
                for ip in range(0, nparticle):
                    if swarm_rad[ip] > Rinner + i * drad and swarm_rad[ip] < Rinner + (i + 1) * drad:
                        swarm_paint[ip] += 1
            for i in [0, 2, 4, 6, 8, 10, 12, 14]:
                dtheta = opening_angle / 16
                for ip in range(0, nparticle):
                    if swarm_theta[ip] > theta_min + i * dtheta and swarm_theta[ip] < theta_min + (i + 1) * dtheta:
                        swarm_paint[ip] += 1
        case _:
            exit("swarm_coordinates_setup: unknown geometry")

    return swarm_active, swarm_x, swarm_z, swarm_r, swarm_t, swarm_id, swarm_iel, swarm_rad, swarm_theta, swarm_paint


###################################################################################################
