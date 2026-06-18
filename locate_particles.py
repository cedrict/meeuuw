###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

from pic_functions import locate_particles___annulus, locate_particles___box

###################################################################################################


def locate_particles(
    geometry: str,
    nparticle: int,
    swarm_active,
    swarm_x,
    swarm_z,
    swarm_rad,
    swarm_theta,
    hx: float,
    hz: float,
    hrad: float,
    htheta: float,
    x_V,
    z_V,
    rad_V,
    theta_V,
    icon_V,
    nelx: int,
    Rinner: float,
):

    match geometry:
        case "box":
            swarm_r, swarm_t, swarm_iel = locate_particles___box(
                nparticle, swarm_active, swarm_x, swarm_z, hx, hz, x_V, z_V, icon_V, nelx
            )
        case "quarter" | "half" | "eighth":
            swarm_r, swarm_t, swarm_iel = locate_particles___annulus(
                nparticle,
                swarm_active,
                swarm_rad,
                swarm_theta,
                hrad,
                htheta,
                rad_V,
                theta_V,
                icon_V,
                nelx,
                Rinner,
            )
        case "annulus":
            exit("locate particles not implemented for annulus geometry")
        case _:
            exit("locate particles not implemented for this geometry")

    print("     -> swarm_r (m,M) %.4e %.4e " % (np.min(swarm_r), np.max(swarm_r)))
    print("     -> swarm_t (m,M) %.4e %.4e " % (np.min(swarm_t), np.max(swarm_t)))
    print("     -> swarm_iel (m,M) %d %d " % (np.min(swarm_iel), np.max(swarm_iel)))

    if np.min(swarm_r) < -1 or np.max(swarm_r) > 1:
        exit("r value out of bounds")
    if np.min(swarm_t) < -1 or np.max(swarm_t) > 1:
        exit("t value out of bounds")

    return swarm_r, swarm_t, swarm_iel


###################################################################################################
