###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

from toolbox import convert_tensor_to_polar_coords, convert_tensor_to_spherical_coords

###################################################################################################


def compute_deviatoric_stress_tensor(
    solve_Stokes,
    geometry,
    x_V,
    theta_V,
    eta_n,
    x_e,
    theta_e,
    eta_e,
    istep,
    nstep,
    every_solution,
    axisymmetric,
    dxx_n,
    dzz_n,
    dxz_n,
    dxx_e,
    dzz_e,
    dxz_e,
    top_Vnodes,
    bot_Vnodes,
    top_element,
    bot_element,
    verbose_output,
    output_folder,
):

    if solve_Stokes:
        tauxx_n = 2 * eta_n * dxx_n
        tauxx_e = 2 * eta_e * dxx_e
        tauzz_n = 2 * eta_n * dzz_n
        tauzz_e = 2 * eta_e * dzz_e
        tauxz_n = 2 * eta_n * dxz_n
        tauxz_e = 2 * eta_e * dxz_e
        taurr_n = 0
        tautt_n = 0
        taurt_n = 0
        taurr_e = 0
        tautt_e = 0
        taurt_e = 0

        if istep % every_solution == 0 or istep == nstep - 1:
            match geometry:
                case "box":
                    np.savetxt(
                        output_folder+"/top/top_tauzz_n_" + str(istep) + ".ascii",
                        np.array([x_V[top_Vnodes], tauzz_n[top_Vnodes]]).T,
                    )
                    np.savetxt(
                        output_folder+"/bottom/bot_tauzz_n_" + str(istep) + ".ascii",
                        np.array([x_V[bot_Vnodes], tauzz_n[bot_Vnodes]]).T,
                    )
                case "quarter" | "half" | "eighth" | "annulus":
                    if axisymmetric:
                        taurr_n, tautt_n, taurt_n = convert_tensor_to_spherical_coords(
                            theta_V, tauxx_n, tauzz_n, tauxz_n
                        )
                        taurr_e, tautt_e, taurt_e = convert_tensor_to_spherical_coords(
                            theta_e, tauxx_e, tauzz_e, tauxz_e
                        )
                    else:
                        taurr_n, tautt_n, taurt_n = convert_tensor_to_polar_coords(theta_V, tauxx_n, tauzz_n, tauxz_n)
                        taurr_e, tautt_e, taurt_e = convert_tensor_to_polar_coords(theta_e, tauxx_e, tauzz_e, tauxz_e)

                    print("     -> taurr_n (m,M) %.3e %.3e " % (np.min(taurr_n), np.max(taurr_n)))
                    print("     -> tautt_n (m,M) %.3e %.3e " % (np.min(tautt_n), np.max(tautt_n)))
                    print("     -> taurt_n (m,M) %.3e %.3e " % (np.min(taurt_n), np.max(taurt_n)))

                    np.savetxt(
                        output_folder+"/top/top_taurr_n_" + str(istep) + ".ascii",
                        np.array([theta_V[top_Vnodes], taurr_n[top_Vnodes]]).T,
                    )
                    np.savetxt(
                        output_folder+"/bottom/bot_taurr_n_" + str(istep) + ".ascii",
                        np.array([theta_V[bot_Vnodes], taurr_n[bot_Vnodes]]).T,
                    )
                    np.savetxt(
                        output_folder+"/top/top_taurr_e_" + str(istep) + ".ascii",
                        np.array([theta_e[top_element], taurr_e[top_element]]).T,
                    )
                    np.savetxt(
                        output_folder+"/bottom/bot_taurr_e_" + str(istep) + ".ascii",
                        np.array([theta_e[bot_element], taurr_e[bot_element]]).T,
                    )

        if verbose_output:
            print("     -> tauxx_n (m,M) %.3e %.3e " % (np.min(tauxx_n), np.max(tauxx_n)))
            print("     -> tauzz_n (m,M) %.3e %.3e " % (np.min(tauzz_n), np.max(tauzz_n)))
            print("     -> tauxz_n (m,M) %.3e %.3e " % (np.min(tauxz_n), np.max(tauxz_n)))
            print("     -> tauxx_e (m,M) %.3e %.3e " % (np.min(tauxx_e), np.max(tauxx_e)))
            print("     -> tauzz_e (m,M) %.3e %.3e " % (np.min(tauzz_e), np.max(tauzz_e)))
            print("     -> tauxz_e (m,M) %.3e %.3e " % (np.min(tauxz_e), np.max(tauxz_e)))

    else:
        tauxx_n = 0
        tauzz_n = 0
        tauxz_n = 0
        tauxx_e = 0
        tauzz_e = 0
        tauxz_e = 0
        taurr_n = 0
        tautt_n = 0
        taurt_n = 0
        taurr_e = 0
        tautt_e = 0
        taurt_e = 0

    return (
        tauxx_n,
        tauzz_n,
        tauxz_n,
        tauxx_e,
        tauzz_e,
        tauxz_e,
        taurr_n,
        tautt_n,
        taurt_n,
        taurr_e,
        tautt_e,
        taurt_e,
    )


###################################################################################################
