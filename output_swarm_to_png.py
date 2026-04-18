###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import matplotlib.pyplot as plt
import numpy as np

###################################################################################################
# TODO: There remains a problem with the fact that the aspect ratio of the domain is 
# not conserved: the colorbar makes the plotting area shrink and alters the true aspect ratio
# I would also be in favour of using the same colorscales as in fieldstone for each field.

def output_swarm_to_png(Lx,Lz,solve_Stokes,solve_T,istep,geometry,nparticle,\
                        swarm_x,swarm_z,swarm_u,swarm_w,swarm_mat,swarm_rho,swarm_eta,\
                        swarm_r,swarm_t,swarm_p,swarm_paint,swarm_exx,swarm_ezz,swarm_exz,swarm_T,\
                        swarm_iel,swarm_hcond,swarm_hcapa,swarm_rad,swarm_theta,swarm_strain,\
                        swarm_F,swarm_sst):

    """
    Args:
        swarm arrays
    Returns:
        png files
    """

    #plt.grid()
    nbdpi=250
    markersize=1

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x,swarm_z,c=swarm_mat,s=markersize)
    plt.gca().set_aspect('equal') ; plt.colorbar(format="%.3e")
    plt.xlim(0,Lx) ; plt.ylim(0,Lz) ; plt.axis('scaled') 
    plt.title('material identity') ; plt.xlabel('x-axis') ; plt.ylabel('y-axis')
    filename='OUTPUT/SWARM/swarm_mat_{:04d}.png'.format(istep)
    plt.savefig(filename, bbox_inches='tight') ; plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x,swarm_z,c=swarm_iel,s=markersize)
    plt.gca().set_aspect('equal') ; plt.colorbar(format="%.3e")
    plt.xlim(0,Lx) ; plt.ylim(0,Lz) ; plt.axis('scaled')
    plt.title('element identity') ; plt.xlabel('x-axis') ; plt.ylabel('y-axis')
    filename='OUTPUT/SWARM/swarm_iel_{:04d}.png'.format(istep)
    plt.savefig(filename, bbox_inches='tight') ; plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x,swarm_z,c=swarm_paint,s=markersize)
    plt.gca().set_aspect('equal') ; plt.colorbar(format="%.3e")
    plt.xlim(0,Lx) ; plt.ylim(0,Lz) ; plt.axis('scaled')
    plt.title('paint') ; plt.xlabel('x-axis') ; plt.ylabel('y-axis')
    filename='OUTPUT/SWARM/swarm_paint_{:04d}.png'.format(istep)
    plt.savefig(filename, bbox_inches='tight') ; plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x,swarm_z,c=swarm_u,s=markersize)
    plt.gca().set_aspect('equal') ; plt.colorbar(format="%.3e")
    plt.xlim(0,Lx) ; plt.ylim(0,Lz) ; plt.axis('scaled')
    plt.title('velocity x-component') ; plt.xlabel('x-axis') ; plt.ylabel('y-axis')
    filename='OUTPUT/SWARM/swarm_u_{:04d}.png'.format(istep)
    plt.savefig(filename, bbox_inches='tight') ; plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x,swarm_z,c=swarm_w,s=markersize)
    plt.gca().set_aspect('equal') ; plt.colorbar(format="%.3e")
    plt.xlim(0,Lx) ; plt.ylim(0,Lz) ; plt.axis('scaled')
    plt.title('velocity z-component') ; plt.xlabel('x-axis') ; plt.ylabel('y-axis')
    filename='OUTPUT/SWARM/swarm_w_{:04d}.png'.format(istep)
    plt.savefig(filename, bbox_inches='tight') ; plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x,swarm_z,c=swarm_rho,s=markersize)
    plt.gca().set_aspect('equal') ; plt.colorbar(format="%.3e")
    plt.xlim(0,Lx) ; plt.ylim(0,Lz) ; plt.axis('scaled')
    plt.title('density') ; plt.xlabel('x-axis') ; plt.ylabel('y-axis')
    filename='OUTPUT/SWARM/swarm_rho_{:04d}.png'.format(istep)
    plt.savefig(filename, bbox_inches='tight') ; plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x,swarm_z,c=np.log10(swarm_eta),s=markersize)
    plt.gca().set_aspect('equal') ; plt.colorbar(format="%.3e")
    plt.xlim(0,Lx) ; plt.ylim(0,Lz) ; plt.axis('scaled')
    plt.title('viscosity') ; plt.xlabel('x-axis') ; plt.ylabel('y-axis')
    filename='OUTPUT/SWARM/swarm_eta_{:04d}.png'.format(istep)
    plt.savefig(filename, bbox_inches='tight') ; plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x,swarm_z,c=swarm_exx,s=markersize)
    plt.gca().set_aspect('equal') ; plt.colorbar(format="%.3e")
    plt.xlim(0,Lx) ; plt.ylim(0,Lz) ; plt.axis('scaled')
    plt.title('strain rate e_xx') ; plt.xlabel('x-axis') ; plt.ylabel('y-axis')
    filename='OUTPUT/SWARM/swarm_exx_{:04d}.png'.format(istep)
    plt.savefig(filename, bbox_inches='tight') ; plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x,swarm_z,c=swarm_ezz,s=markersize)
    plt.gca().set_aspect('equal') ; plt.colorbar(format="%.3e")
    plt.xlim(0,Lx) ; plt.ylim(0,Lz) ; plt.axis('scaled')
    plt.title('strain rate e_zz') ; plt.xlabel('x-axis') ; plt.ylabel('y-axis')
    filename='OUTPUT/SWARM/swarm_ezz_{:04d}.png'.format(istep)
    plt.savefig(filename, bbox_inches='tight') ; plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x,swarm_z,c=swarm_exz,s=markersize)
    plt.gca().set_aspect('equal') ; plt.colorbar(format="%.3e")
    plt.xlim(0,Lx) ; plt.ylim(0,Lz) ; plt.axis('scaled')
    plt.title('strain rate e_xz') ; plt.xlabel('x-axis') ; plt.ylabel('y-axis')
    filename='OUTPUT/SWARM/swarm_exz_{:04d}.png'.format(istep)
    plt.savefig(filename, bbox_inches='tight') ; plt.close()

    plt.figure(dpi=nbdpi)
    plt.scatter(swarm_x,swarm_z,c=swarm_p,s=markersize)
    plt.gca().set_aspect('equal') ; plt.colorbar(format="%.3e")
    plt.xlim(0,Lx) ; plt.ylim(0,Lz) ; plt.axis('scaled')
    plt.title('pressure') ; plt.xlabel('x-axis') ; plt.ylabel('y-axis')
    filename='OUTPUT/SWARM/swarm_p_{:04d}.png'.format(istep)
    plt.savefig(filename, bbox_inches='tight') ; plt.close()

    if solve_T:

       plt.figure(dpi=nbdpi)
       plt.scatter(swarm_x,swarm_z,c=swarm_T,s=markersize)
       plt.gca().set_aspect('equal') ; plt.colorbar(format="%.3e")
       plt.xlim(0,Lx) ; plt.ylim(0,Lz) ; plt.axis('scaled')
       plt.title('temperature') ; plt.xlabel('x-axis') ; plt.ylabel('y-axis')
       filename='OUTPUT/SWARM/swarm_T_{:04d}.png'.format(istep)
       plt.savefig(filename, bbox_inches='tight') ; plt.close()

       plt.figure(dpi=nbdpi)
       plt.scatter(swarm_x,swarm_z,c=swarm_hcond,s=markersize)
       plt.gca().set_aspect('equal') ; plt.colorbar(format="%.3e")
       plt.xlim(0,Lx) ; plt.ylim(0,Lz) ; plt.axis('scaled')
       plt.title('heat conductivity') ; plt.xlabel('x-axis') ; plt.ylabel('y-axis')
       filename='OUTPUT/SWARM/swarm_hcond_{:04d}.png'.format(istep)
       plt.savefig(filename, bbox_inches='tight') ; plt.close()

       plt.figure(dpi=nbdpi)
       plt.scatter(swarm_x,swarm_z,c=swarm_hcapa,s=markersize)
       plt.gca().set_aspect('equal') ; plt.colorbar(format="%.3e")
       plt.xlim(0,Lx) ; plt.ylim(0,Lz) ; plt.axis('scaled')
       plt.title('heat capacity') ; plt.xlabel('x-axis') ; plt.ylabel('y-axis')
       filename='OUTPUT/SWARM/swarm_hcapa_{:04d}.png'.format(istep)
       plt.savefig(filename, bbox_inches='tight') ; plt.close()

###################################################################################################
