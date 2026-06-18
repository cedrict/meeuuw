###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numba
import numpy as np

###################################################################################################
# compute avrg temperature, viscosity, velocity profiles | not the most elegant but works
###################################################################################################


@numba.njit
def compute_avrg_profiles(geometry, nnx, nnz, T, rho_n, eta_n, u, w, q, z_V, rad_V):

    q_profile = np.zeros(nnz, dtype=np.float64)
    T_profile = np.zeros(nnz, dtype=np.float64)
    vel_profile = np.zeros(nnz, dtype=np.float64)
    eta_profile = np.zeros(nnz, dtype=np.float64)
    rho_profile = np.zeros(nnz, dtype=np.float64)
    coord_profile = np.zeros(nnz, dtype=np.float64)

    counter = 0
    for j in range(0, nnz):
        if geometry == "box":
            coord_profile[j] = z_V[counter]
        else:
            coord_profile[j] = rad_V[counter]

        for i in range(0, nnx):
            q_profile[j] += q[counter] / nnx
            T_profile[j] += T[counter] / nnx
            eta_profile[j] += eta_n[counter] / nnx
            rho_profile[j] += rho_n[counter] / nnx
            vel_profile[j] += np.sqrt(u[counter] ** 2 + w[counter] ** 2) / nnx
            counter += 1

    return T_profile, vel_profile, eta_profile, q_profile, rho_profile, coord_profile


###################################################################################################
