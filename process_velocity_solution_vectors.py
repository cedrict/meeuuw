###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

from remove_net_rotation import remove_net_rotation

###################################################################################################


def process_velocity_solution_vectors(
    geometry,
    u,
    w,
    x_V,
    z_V,
    x_P,
    z_P,
    top_free_slip,
    bot_free_slip,
    top_Vnodes,
    bot_Vnodes,
    bc_fix_V,
    vel_scale,
    vel_unit,
    geological_time,
    time_scale,
    istep,
    nstep,
    nq_per_element,
    nel,
    icon_V,
    xq,
    zq,
    nn_V,
    N_V,
    JxWq,
    rad_V,
    theta_V,
    rad_P,
    theta_P,
    debug_nan,
    debug_ascii,
    every_solution,
    vstats_file,
):

    if debug_nan and np.isnan(np.sum(u)):
        exit("nan found in u")
    if debug_nan and np.isnan(np.sum(w)):
        exit("nan found in w")

    match geometry:
        case "quarter" | "half" | "eighth" | "annulus":
            if top_free_slip:
                for i in range(0, nn_V):
                    if top_Vnodes[i] and (not bc_fix_V[2 * i]) and (not bc_fix_V[2 * i + 1]):
                        ui = np.cos(theta_V[i]) * u[i] - np.sin(theta_V[i]) * w[i]
                        wi = np.sin(theta_V[i]) * u[i] + np.cos(theta_V[i]) * w[i]
                        u[i] = ui
                        w[i] = wi
            if bot_free_slip:
                for i in range(0, nn_V):
                    if bot_Vnodes[i] and (not bc_fix_V[2 * i]) and (not bc_fix_V[2 * i + 1]):
                        ui = np.cos(theta_V[i]) * u[i] - np.sin(theta_V[i]) * w[i]
                        wi = np.sin(theta_V[i]) * u[i] + np.cos(theta_V[i]) * w[i]
                        u[i] = ui
                        w[i] = wi

    # In the case the domain is an annulus and free slip is prescribed
    # on both boundaries then the rotational nullspace must be removed

    if geometry == "annulus" and top_free_slip and bot_free_slip:
        print("     -> u (m,M) %.3e %.3e %s" % (np.min(u) / vel_scale, np.max(u) / vel_scale, vel_unit))
        print("     -> w (m,M) %.3e %.3e %s" % (np.min(w) / vel_scale, np.max(w) / vel_scale, vel_unit))
        u, w, omega_z = remove_net_rotation(nq_per_element, nel, icon_V, xq, zq, u, w, nn_V, N_V, x_V, z_V, JxWq)
        print("     -> ang. momentum omega_z %e " % omega_z)

    vel = np.sqrt(u**2 + w**2)

    if istep % every_solution == 0 or istep == nstep - 1:
        match geometry:
            case "box":
                np.savetxt(
                    "OUTPUT/top/top_vel_" + str(istep) + ".ascii",
                    np.array([x_V[top_Vnodes], vel[top_Vnodes]]).T,
                )
                np.savetxt(
                    "OUTPUT/bottom/bot_vel_" + str(istep) + ".ascii",
                    np.array([x_V[bot_Vnodes], vel[bot_Vnodes]]).T,
                )
            case "eighth" | "quarter" | "half" | "annulus":
                np.savetxt(
                    "OUTPUT/top/top_vel_" + str(istep) + ".ascii",
                    np.array([theta_V[top_Vnodes], vel[top_Vnodes]]).T,
                )
                np.savetxt(
                    "OUTPUT/bottom/bot_vel_" + str(istep) + ".ascii",
                    np.array([theta_V[bot_Vnodes], vel[bot_Vnodes]]).T,
                )
            case _:
                raise ValueError("process_velocity_solution_vectors: unknown geometry")

    print("     -> u (m,M) %.3e %.3e %s" % (np.min(u) / vel_scale, np.max(u) / vel_scale, vel_unit))
    print("     -> w (m,M) %.3e %.3e %s" % (np.min(w) / vel_scale, np.max(w) / vel_scale, vel_unit))

    vstats_file.write(
        "%.3e %.3e %.3e %.3e %.3e\n"
        % (
            geological_time / time_scale,
            np.min(u) / vel_scale,
            np.max(u) / vel_scale,
            np.min(w) / vel_scale,
            np.max(w) / vel_scale,
        )
    )
    vstats_file.flush()

    if debug_ascii:
        np.savetxt(
            "DEBUG/velocity.ascii",
            np.array([x_V, z_V, u, w, vel, rad_V, theta_V]).T,
            header="# x,z,u,w,vel,rad,theta",
        )

    return u, w, vel


###################################################################################################
