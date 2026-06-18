###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################


def compute_domain_volume(geometry: str, Lx: float, Lz: float, Rinner: float, Router: float) -> float:
    """

    Args:
        geometry: geometry of the computational domain
        Lx,Lz: domain size of Cartesian box geometry
        Rinner,Router: inner and outer radius of domain

    Returns:
        the analytical volume of the computational domain
    """

    match geometry:
        case "box":
            volume = Lx * Lz
        case "eighth":
            volume = np.pi * (Router**2 - Rinner**2) / 8
        case "quarter":
            volume = np.pi * (Router**2 - Rinner**2) / 4
        case "half":
            volume = np.pi * (Router**2 - Rinner**2) / 2
        case "annulus":
            volume = np.pi * (Router**2 - Rinner**2)
        case _:
            raise ValueError("compute_domain_volume: unknown geometry")

    return volume


###################################################################################################
