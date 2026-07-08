###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

from toolbox import convert_tensor_to_polar_coords, convert_tensor_to_spherical_coords

###################################################################################################


def compute_full_stress_tensor(
    solve_Stokes,
    geometry,
    x_V,
    theta_V,
    q,
    x_e,
    theta_e,
    p_e,
    istep,
    nstep,
    every_solution,
    axisymmetric,
    tauxx_n,
    tauzz_n,
    tauxz_n,
    tauxx_e,
    tauzz_e,
    tauxz_e,
    top_Vnodes,
    bot_Vnodes,
    top_element,
    bot_element,
    output_folder,
):

    if solve_Stokes:
        sigmaxx_n = -q + tauxx_n
        sigmaxx_e = -p_e + tauxx_e
        sigmazz_n = -q + tauzz_n
        sigmazz_e = -p_e + tauzz_e
        sigmaxz_n = tauxz_n
        sigmaxz_e = tauxz_e
        sigmarr_n = 0
        sigmatt_n = 0
        sigmart_n = 0
        sigmarr_e = 0
        sigmatt_e = 0
        sigmart_e = 0

        if istep % every_solution == 0 or istep == nstep - 1:
            match geometry:
                case "box":
                    np.savetxt(
                        output_folder+"/top/top_sigmazz_n_" + str(istep) + ".ascii",
                        np.array([x_V[top_Vnodes], sigmazz_n[top_Vnodes]]).T,
                    )
                    np.savetxt(
                        output_folder+"/bottom/bot_sigmazz_n_" + str(istep) + ".ascii",
                        np.array([x_V[bot_Vnodes], sigmazz_n[bot_Vnodes]]).T,
                    )
                    np.savetxt(
                        output_folder+"/top/top_sigmazz_e_" + str(istep) + ".ascii",
                        np.array([x_e[top_element], sigmazz_e[top_element]]).T,
                    )
                    np.savetxt(
                        output_folder+"/bottom/bot_sigmazz_e_" + str(istep) + ".ascii",
                        np.array([x_e[bot_element], sigmazz_e[bot_element]]).T,
                    )
                case "quarter" | "half" | "eighth" | "annulus":
                    if axisymmetric:
                        sigmarr_n, sigmatt_n, sigmart_n = convert_tensor_to_spherical_coords(
                            theta_V, sigmaxx_n, sigmazz_n, sigmaxz_n
                        )
                        sigmarr_e, sigmatt_e, sigmart_e = convert_tensor_to_spherical_coords(
                            theta_e, sigmaxx_e, sigmazz_e, sigmaxz_e
                        )
                    else:
                        sigmarr_n, sigmatt_n, sigmart_n = convert_tensor_to_polar_coords(
                            theta_V, sigmaxx_n, sigmazz_n, sigmaxz_n
                        )
                        sigmarr_e, sigmatt_e, sigmart_e = convert_tensor_to_polar_coords(
                            theta_e, sigmaxx_e, sigmazz_e, sigmaxz_e
                        )
                    np.savetxt(
                        output_folder+"/top/top_sigmarr_n_" + str(istep) + ".ascii",
                        np.array([theta_V[top_Vnodes], sigmarr_n[top_Vnodes]]).T,
                    )
                    np.savetxt(
                        output_folder+"/bottom/bot_sigmarr_n_" + str(istep) + ".ascii",
                        np.array([theta_V[bot_Vnodes], sigmarr_n[bot_Vnodes]]).T,
                    )
                    np.savetxt(
                        output_folder+"/top/top_sigmarr_e_" + str(istep) + ".ascii",
                        np.array([theta_e[top_element], sigmarr_e[top_element]]).T,
                    )
                    np.savetxt(
                        output_folder+"/bottom/bot_sigmarr_e_" + str(istep) + ".ascii",
                        np.array([theta_e[bot_element], sigmarr_e[bot_element]]).T,
                    )

    else:
        sigmaxx_n = 0
        sigmazz_n = 0
        sigmaxz_n = 0
        sigmaxx_e = 0
        sigmazz_e = 0
        sigmaxz_e = 0
        sigmarr_n = 0
        sigmatt_n = 0
        sigmart_n = 0
        sigmarr_e = 0
        sigmatt_e = 0
        sigmart_e = 0

    return (
        sigmaxx_n,
        sigmazz_n,
        sigmaxz_n,
        sigmaxx_e,
        sigmazz_e,
        sigmaxz_e,
        sigmarr_n,
        sigmatt_n,
        sigmart_n,
        sigmarr_e,
        sigmatt_e,
        sigmart_e,
    )


###################################################################################################
