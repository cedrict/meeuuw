###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import os

import numpy as np

###################################################################################################


def output_swarm_to_ascii(
    Lx,
    Lz,
    solve_Stokes,
    solve_T,
    istep,
    geometry,
    nparticle,
    swarm_active,
    swarm_x,
    swarm_z,
    swarm_u,
    swarm_w,
    swarm_wf,
    swarm_rho,
    swarm_eta,
    swarm_r,
    swarm_t,
    swarm_p,
    swarm_paint,
    swarm_exx,
    swarm_ezz,
    swarm_exz,
    swarm_T,
    swarm_iel,
    swarm_hcond,
    swarm_hcapa,
    swarm_alpha,
    swarm_rad,
    swarm_theta,
    swarm_strain,
    swarm_F,
    swarm_sst,
    output_folder,
):
    """
    Args:
    Returns:
    """

    filename = output_folder+"/SWARM/swarm_{:04d}.ascii".format(istep)
    np.savetxt(
        filename,
        np.array(
            [
                swarm_x[swarm_active],
                swarm_z[swarm_active],
                swarm_u[swarm_active],
                swarm_w[swarm_active],
                swarm_rho[swarm_active],
                swarm_eta[swarm_active],
                swarm_paint[swarm_active],
                swarm_exx[swarm_active],
                swarm_ezz[swarm_active],
                swarm_exz[swarm_active],
            ]
        ).T,
        header="#x,z,u,w,T,mat,rho,eta,paint,exx,ezz,exz",
        fmt="%.3e",
    )

    #######################################################

    #if False:
    #    gnuplot_file = open("OUTPUT/SWARM/gnuplot_script", "w")
    #    gnuplot_file.write("set term png size %d %d font 'Times,8pt' \n" % (int(Lx / Lz * 400), 400))

    #    filename2 = "swarm_{:04d}.ascii".format(istep)

    #    filename = "gnuplot_swarm_u_{:04d}.png".format(istep)
    #    gnuplot_file.write("set title 'velocity x-component' \n")
    #    gnuplot_file.write("set output '" + filename + "' \n")
    #    gnuplot_file.write("plot[][] './OUTPUT/SWARM/" + filename2 + "' u 1:2:3 palette w d notitle \n")

    #    filename = "gnuplot_swarm_w_{:04d}.png".format(istep)
    #    gnuplot_file.write("set title 'velocity z-component' \n")
    #    gnuplot_file.write("set output '" + filename + "' \n")
    #    gnuplot_file.write("plot[][] './OUTPUT/SWARM/" + filename2 + "' u 1:2:4 palette w d notitle \n")

        # filename='gnuplot_swarm_mat_{:04d}.png'.format(istep)
        # gnuplot_file.write("set title 'material' \n")
        # gnuplot_file.write("set output '"+filename+"' \n")
        # gnuplot_file.write("plot[][] './OUTPUT/SWARM/"+filename2+"' u 1:2:5 palette w d notitle \n")

    #    filename = "gnuplot_swarm_rho_{:04d}.png".format(istep)
    #    gnuplot_file.write("set title 'density' \n")
    #    gnuplot_file.write("set output '" + filename + "' \n")
    #    gnuplot_file.write("plot[][] './OUTPUT/SWARM/" + filename2 + "' u 1:2:6 palette w d notitle \n")

    #    gnuplot_file.close()

    #    os.system("gnuplot ./OUTPUT/SWARM/gnuplot_script")
    #    os.system("mv gnuplot_swarm*.png OUTPUT/SWARM/")


###################################################################################################
