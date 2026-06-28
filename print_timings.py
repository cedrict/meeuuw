###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################


def print_timings(iloop, timings, duration):
    """

    Args:
        a:
        b:

    Returns:
        bla
    """

    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")
    print(
        "build FE matrix V: %8.3f s      (%.3f s per call) | %5.2f percent"
        % (timings[1], timings[1] / (iloop + 1), timings[1] / duration * 100)
    )
    print(
        "solve system V: %8.3f s         (%.3f s per call) | %5.2f percent"
        % (timings[2], timings[2] / (iloop + 1), timings[2] / duration * 100)
    )
    print(
        "build FE matrix T: %8.3f s      (%.3f s per call) | %5.2f percent"
        % (timings[4], timings[4] / (iloop + 1), timings[4] / duration * 100)
    )
    print(
        "solve system T: %8.3f s         (%.3f s per call) | %5.2f percent"
        % (timings[5], timings[5] / (iloop + 1), timings[5] / duration * 100)
    )
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print(
        "compute plith: %8.3f s          (%.3f s per call) | %5.2f percent"
        % (timings[28], timings[28] / (iloop + 1), timings[28] / duration * 100)
    )
    print(
        "comp. glob quantities: %8.3f s  (%.3f s per call) | %5.2f percent"
        % (timings[6], timings[6] / (iloop + 1), timings[6] / duration * 100)
    )
    print(
        "comp. nodal p: %8.3f s          (%.3f s per call) | %5.2f percent"
        % (timings[3], timings[3] / (iloop + 1), timings[3] / duration * 100)
    )
    print(
        "comp. nodal sr: %8.3f s         (%.3f s per call) | %5.2f percent"
        % (timings[11], timings[11] / (iloop + 1), timings[11] / duration * 100)
    )
    print(
        "comp. nodal stress: %8.3f s     (%.3f s per call) | %5.2f percent"
        % (timings[27], timings[27] / (iloop + 1), timings[27] / duration * 100)
    )
    print(
        "comp. nodal heat flux: %8.3f s  (%.3f s per call) | %5.2f percent"
        % (timings[7], timings[7] / (iloop + 1), timings[7] / duration * 100)
    )
    print(
        "comp. eltal sr: %8.3f s         (%.3f s per call) | %5.2f percent"
        % (timings[29], timings[29] / (iloop + 1), timings[29] / duration * 100)
    )
    print(
        "comp. T profile: %8.3f s        (%.3f s per call) | %5.2f percent"
        % (timings[9], timings[9] / (iloop + 1), timings[9] / duration * 100)
    )
    print(
        "comp. nodal press grad: %8.3f s (%.3f s per call) | %5.2f percent"
        % (timings[8], timings[8] / (iloop + 1), timings[8] / duration * 100)
    )
    print(
        "normalise pressure: %8.3f s     (%.3f s per call) | %5.2f percent"
        % (timings[12], timings[12] / (iloop + 1), timings[12] / duration * 100)
    )
    print(
        "compute el pressure: %8.3f s    (%.3f s per call) | %5.2f percent"
        % (timings[33], timings[33] / (iloop + 1), timings[33] / duration * 100)
    )
    print(
        "advect particles: %8.3f s       (%.3f s per call) | %5.2f percent"
        % (timings[13], timings[13] / (iloop + 1), timings[13] / duration * 100)
    )
    print(
        "split solution: %8.3f s         (%.3f s per call) | %5.2f percent"
        % (timings[14], timings[14] / (iloop + 1), timings[14] / duration * 100)
    )
    print(
        "material model on ptcls: %8.3fs (%.3f s per call) | %5.2f percent"
        % (timings[15], timings[15] / (iloop + 1), timings[15] / duration * 100)
    )
    print(
        "locate particles: %8.3f s       (%.3f s per call) | %5.2f percent"
        % (timings[16], timings[16] / (iloop + 1), timings[16] / duration * 100)
    )
    print(
        "comp eltal rho,eta: %8.3f s     (%.3f s per call) | %5.2f percent"
        % (timings[17], timings[17] / (iloop + 1), timings[17] / duration * 100)
    )
    print(
        "comp nodal rho,eta: %8.3f s     (%.3f s per call) | %5.2f percent"
        % (timings[18], timings[18] / (iloop + 1), timings[18] / duration * 100)
    )
    print(
        "compute dyn topo: %8.3f s       (%.3f s per call) | %5.2f percent"
        % (timings[26], timings[26] / (iloop + 1), timings[26] / duration * 100)
    )
    print(
        "comp timestep: %8.3f s          (%.3f s per call) | %5.2f percent"
        % (timings[19], timings[19] / (iloop + 1), timings[19] / duration * 100)
    )
    print(
        "project fields on qpts: %8.3f s (%.3f s per call) | %5.2f percent"
        % (timings[21], timings[21] / (iloop + 1), timings[21] / duration * 100)
    )
    print(
        "compute gravity: %8.3f s        (%.3f s per call) | %5.2f percent"
        % (timings[23], timings[23] / (iloop + 1), timings[23] / duration * 100)
    )
    print(
        "interp sr,p,T on ptcls: %8.3f s (%.3f s per call) | %5.2f percent"
        % (timings[24], timings[24] / (iloop + 1), timings[24] / duration * 100)
    )
    print(
        "least squares fit: %8.3f s      (%.3f s per call) | %5.2f percent"
        % (timings[25], timings[25] / (iloop + 1), timings[25] / duration * 100)
    )
    print(
        "compute rho_profile: %8.3f s    (%.3f s per call) | %5.2f percent"
        % (timings[30], timings[30] / (iloop + 1), timings[30] / duration * 100)
    )
    print(
        "remove rho_profile:  %8.3f s    (%.3f s per call) | %5.2f percent"
        % (timings[31], timings[31] / (iloop + 1), timings[31] / duration * 100)
    )
    print(
        "vel to polar coords: %8.3f s    (%.3f s per call) | %5.2f percent"
        % (timings[32], timings[32] / (iloop + 1), timings[32] / duration * 100)
    )
    print(
        "process u,v vectors: %8.3f s    (%.3f s per call) | %5.2f percent"
        % (timings[37], timings[37] / (iloop + 1), timings[37] / duration * 100)
    )
    print(
        "assess_steady_state: %8.3f s    (%.3f s per call) | %5.2f percent"
        % (timings[38], timings[38] / (iloop + 1), timings[38] / duration * 100)
    )
    print(
        "population_control:  %8.3f s    (%.3f s per call) | %5.2f percent"
        % (timings[39], timings[39] / (iloop + 1), timings[39] / duration * 100)
    )

    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print(
        "output solution to vtu: %8.3f s (%.3f s per call) | %5.2f percent"
        % (timings[10], timings[10] / (iloop + 1), timings[10] / duration * 100)
    )
    print(
        "output swarm to vtu: %8.3f s    (%.3f s per call) | %5.2f percent"
        % (timings[20], timings[20] / (iloop + 1), timings[20] / duration * 100)
    )
    print(
        "output qpts to vtu: %8.3f s     (%.3f s per call) | %5.2f percent"
        % (timings[22], timings[22] / (iloop + 1), timings[22] / duration * 100)
    )
    print(
        "output solution to png: %8.3f s (%.3f s per call) | %5.2f percent"
        % (timings[34], timings[34] / (iloop + 1), timings[34] / duration * 100)
    )
    print(
        "output swarm to png: %8.3f s    (%.3f s per call) | %5.2f percent"
        % (timings[35], timings[35] / (iloop + 1), timings[35] / duration * 100)
    )
    print(
        "output swarm to ascii: %8.3f s  (%.3f s per call) | %5.2f percent"
        % (timings[36], timings[36] / (iloop + 1), timings[36] / duration * 100)
    )
    print("----------------------------------------------------------------------")
    print("compute time per timestep: %.2f" % (duration / (iloop + 1)))
    print("----------------------------------------------------------------------")

    return 1


###################################################################################################
