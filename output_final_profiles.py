###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################


def output_final_profiles(
    x_V,
    z_V,
    u,
    w,
    q,
    T,
    rho_n,
    eta_n,
    middleV_nodes,
    middleH_nodes,
    x_e,
    z_e,
    p_e,
    rho_e,
    eta_e,
    middleV_element,
    middleH_element,
    output_folder,
):

    #######################################################
    np.savetxt(
        output_folder+"/profiles/profile_vertical.ascii",
        np.array(
            [
                z_V[middleV_nodes],
                u[middleV_nodes],
                w[middleV_nodes],
                q[middleV_nodes],
                T[middleV_nodes],
                rho_n[middleV_nodes],
                eta_n[middleV_nodes],
            ]
        ).T,
        header="#z u w q T rho eta",
    )

    #######################################################
    np.savetxt(
        output_folder+"/profiles/profile_horizontal.ascii",
        np.array(
            [
                x_V[middleH_nodes],
                u[middleH_nodes],
                w[middleH_nodes],
                q[middleH_nodes],
                T[middleH_nodes],
                rho_n[middleH_nodes],
                eta_n[middleH_nodes],
            ]
        ).T,
        header="#x u w q T rho eta",
    )

    #######################################################
    np.savetxt(
        output_folder+"/profiles/profile_vertical_e.ascii",
        np.array(
            [
                z_e[middleV_element],
                p_e[middleV_element],
                rho_e[middleV_element],
                eta_e[middleV_element],
            ]
        ).T,
        header="#z p rho eta",
    )

    #######################################################
    np.savetxt(
        output_folder+"/profiles/profile_horizontal_e.ascii",
        np.array(
            [
                x_e[middleH_element],
                p_e[middleH_element],
                rho_e[middleH_element],
                eta_e[middleH_element],
            ]
        ).T,
        header="#x p rho eta",
    )

    return


###################################################################################################
