###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import matplotlib.pyplot as plt
import numpy as np

###################################################################################################
# TODO: There remains a problem with the fact that the aspect ratio of the domain is
# not conserved: the colorbar makes the plotting area shrink and alters the true aspect ratio
# I would also be in favour of using the same colorscales as in fieldstone for each field.


def output_swarm_to_png(
    Lx,
    Lz,
    solve_Stokes,
    solve_T,
    istep,
    geometry,
    nparticle,
    nmat,
    material_names,
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
        swarm arrays
    Returns:
        png files
    """

    # plt.grid()
    nbdpi = 250
    markersize = 1

    # plt.figure(dpi=nbdpi)
    # plt.scatter(swarm_x,swarm_z,c=swarm_mat,s=markersize)
    # plt.gca().set_aspect('equal') ; plt.colorbar(format="%.3e")
    # plt.xlim(0,Lx) ; plt.ylim(0,Lz) ; plt.axis('scaled')
    # plt.title('material identity') ; plt.xlabel('x-axis') ; plt.ylabel('y-axis')
    # filename='/SWARM/swarm_mat_{:04d}.png'.format(istep)
    # plt.savefig(filename, bbox_inches='tight') ; plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x[swarm_active], swarm_z[swarm_active], c=swarm_iel[swarm_active], s=markersize)
    plt.gca().set_aspect("equal")
    plt.colorbar(format="%.3e")
    plt.xlim(0, Lx)
    plt.ylim(0, Lz)
    plt.axis("scaled")
    plt.title("element identity")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    
    filename = output_folder+"/SWARM/swarm_iel_{:04d}.png".format(istep)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x[swarm_active], swarm_z[swarm_active], c=swarm_paint[swarm_active], s=markersize)
    plt.gca().set_aspect("equal")
    plt.colorbar(format="%.3e")
    plt.xlim(0, Lx)
    plt.ylim(0, Lz)
    plt.axis("scaled")
    plt.title("paint")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    filename = output_folder+"/SWARM/swarm_paint_{:04d}.png".format(istep)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x[swarm_active], swarm_z[swarm_active], c=swarm_u[swarm_active], s=markersize)
    plt.gca().set_aspect("equal")
    plt.colorbar(format="%.3e")
    plt.xlim(0, Lx)
    plt.ylim(0, Lz)
    plt.axis("scaled")
    plt.title("velocity x-component")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    filename = output_folder+"/SWARM/swarm_u_{:04d}.png".format(istep)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x[swarm_active], swarm_z[swarm_active], c=swarm_w[swarm_active], s=markersize)
    plt.gca().set_aspect("equal")
    plt.colorbar(format="%.3e")
    plt.xlim(0, Lx)
    plt.ylim(0, Lz)
    plt.axis("scaled")
    plt.title("velocity z-component")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    filename = output_folder+"/SWARM/swarm_w_{:04d}.png".format(istep)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x[swarm_active], swarm_z[swarm_active], c=swarm_rho[swarm_active], s=markersize)
    plt.gca().set_aspect("equal")
    plt.colorbar(format="%.3e")
    plt.xlim(0, Lx)
    plt.ylim(0, Lz)
    plt.axis("scaled")
    plt.title("density")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    filename = output_folder+"/SWARM/swarm_rho_{:04d}.png".format(istep)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x[swarm_active], swarm_z[swarm_active], c=np.log10(swarm_eta[swarm_active]), s=markersize)
    plt.gca().set_aspect("equal")
    plt.colorbar(format="%.3e")
    plt.xlim(0, Lx)
    plt.ylim(0, Lz)
    plt.axis("scaled")
    plt.title("viscosity")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    filename = output_folder+"/SWARM/swarm_eta_{:04d}.png".format(istep)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x[swarm_active], swarm_z[swarm_active], c=swarm_exx[swarm_active], s=markersize)
    plt.gca().set_aspect("equal")
    plt.colorbar(format="%.3e")
    plt.xlim(0, Lx)
    plt.ylim(0, Lz)
    plt.axis("scaled")
    plt.title("strain rate e_xx")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    filename = output_folder+"/SWARM/swarm_exx_{:04d}.png".format(istep)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x[swarm_active], swarm_z[swarm_active], c=swarm_ezz[swarm_active], s=markersize)
    plt.gca().set_aspect("equal")
    plt.colorbar(format="%.3e")
    plt.xlim(0, Lx)
    plt.ylim(0, Lz)
    plt.axis("scaled")
    plt.title("strain rate e_zz")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    filename = output_folder+"/SWARM/swarm_ezz_{:04d}.png".format(istep)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x[swarm_active], swarm_z[swarm_active], c=swarm_exz[swarm_active], s=markersize)
    plt.gca().set_aspect("equal")
    plt.colorbar(format="%.3e")
    plt.xlim(0, Lx)
    plt.ylim(0, Lz)
    plt.axis("scaled")
    plt.title("strain rate e_xz")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    filename = output_folder+"/SWARM/swarm_exz_{:04d}.png".format(istep)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x[swarm_active], swarm_z[swarm_active], c=swarm_p[swarm_active], s=markersize)
    plt.gca().set_aspect("equal")
    plt.colorbar(format="%.3e")
    plt.xlim(0, Lx)
    plt.ylim(0, Lz)
    plt.axis("scaled")
    plt.title("pressure")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    filename = output_folder+"/SWARM/swarm_p_{:04d}.png".format(istep)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    if solve_T:
        plt.figure(dpi=nbdpi)
        plt.scatter(swarm_x[swarm_active], swarm_z[swarm_active], c=swarm_T[swarm_active], s=markersize)
        plt.gca().set_aspect("equal")
        plt.colorbar(format="%.3e")
        plt.xlim(0, Lx)
        plt.ylim(0, Lz)
        plt.axis("scaled")
        plt.title("temperature")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        filename = output_folder+"/SWARM/swarm_T_{:04d}.png".format(istep)
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

        plt.figure(dpi=nbdpi)
        plt.scatter(swarm_x[swarm_active], swarm_z[swarm_active], c=swarm_hcond[swarm_active], s=markersize)
        plt.gca().set_aspect("equal")
        plt.colorbar(format="%.3e")
        plt.xlim(0, Lx)
        plt.ylim(0, Lz)
        plt.axis("scaled")
        plt.title("heat conductivity")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        filename = output_folder+"/SWARM/swarm_hcond_{:04d}.png".format(istep)
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

        plt.figure(dpi=nbdpi)
        plt.scatter(swarm_x[swarm_active], swarm_z[swarm_active], c=swarm_hcapa[swarm_active], s=markersize)
        plt.gca().set_aspect("equal")
        plt.colorbar(format="%.3e")
        plt.xlim(0, Lx)
        plt.ylim(0, Lz)
        plt.axis("scaled")
        plt.title("heat capacity")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        filename = output_folder+"/SWARM/swarm_hcapa_{:04d}.png".format(istep)
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


###################################################################################################
