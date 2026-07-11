###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

from toolbox import effective

###################################################################################################


def output_swarm_to_vtu(
    solve_Stokes,
    use_melting,
    TKelvin,
    istep,
    geometry,
    nparticle,
    nmat,
    solve_T,
    vel_scale,
    material_names,
    swarm_active,
    swarm_id,
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
    swarm_mechanism,
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

    nparticle_active=np.sum(swarm_active)

    debug_swarm = False

    filename = output_folder+"/SWARM/swarm_{:04d}.vtu".format(istep)
    vtufile = open(filename, "w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" % (nparticle_active, nparticle_active))
    #####
    vtufile.write("<Points> \n")
    # --
    swarm_X = np.zeros(nparticle_active, dtype=np.float64)
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    coords = np.array([swarm_x[swarm_active], swarm_X, swarm_z[swarm_active]]).T
    coords.tofile(vtufile, sep=" ")

    vtufile.write("</DataArray>\n")
    # --
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    # --
    if debug_swarm and solve_Stokes:
        vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Velocity' Format='ascii'> \n")
        coords = np.array([swarm_u[swarm_active], swarm_X, swarm_w[swarm_active]]).T
        coords.tofile(vtufile, sep=" ")
        vtufile.write("</DataArray>\n")
        # --
        vtufile.write("<DataArray type='Float32' Name='exx' Format='binary'> \n")
        swarm_exx[swarm_active].tofile(vtufile, sep=" ", format="%.4e")
        vtufile.write("</DataArray>\n")
        # --
        vtufile.write("<DataArray type='Float32' Name='ezz' Format='binary'> \n")
        swarm_ezz[swarm_active].tofile(vtufile, sep=" ", format="%.4e")
        vtufile.write("</DataArray>\n")
        # --
        vtufile.write("<DataArray type='Float32' Name='exz' Format='binary'> \n")
        swarm_exz[swarm_active].tofile(vtufile, sep=" ", format="%.4e")
        vtufile.write("</DataArray>\n")
        # --
        vtufile.write("<DataArray type='Float32' Name='Pressure' Format='binary'> \n")
        swarm_p[swarm_active].tofile(vtufile, sep=" ", format="%.5e")
        vtufile.write("</DataArray>\n")
        # --
        vtufile.write("<DataArray type='Float32' Name='r' Format='binary'> \n")
        swarm_r[swarm_active].tofile(vtufile, sep=" ", format="%.3e")
        vtufile.write("</DataArray>\n")
        # --
        vtufile.write("<DataArray type='Float32' Name='t' Format='binary'> \n")
        swarm_t[swarm_active].tofile(vtufile, sep=" ", format="%.3e")
        vtufile.write("</DataArray>\n")
        # --
        vtufile.write("<DataArray type='Int32' Name='iel' Format='binary'> \n")
        swarm_iel[swarm_active].tofile(vtufile, sep=" ", format="%d")
        vtufile.write("</DataArray>\n")
    # --
    for imat in range(0, nmat):
        vtufile.write("<DataArray type='Float32' Name='" + material_names[imat] + "' Format='binary'> \n")
        swarm_wf[imat, :][swarm_active].tofile(vtufile, sep=" ", format="%.3e")
        vtufile.write("</DataArray>\n")
    # --
    ee = effective(swarm_exx, swarm_ezz, swarm_exz)
    vtufile.write("<DataArray type='Float32' Name='e' Format='binary'> \n")
    ee.tofile(vtufile, sep=" ", format="%.4e")
    vtufile.write("</DataArray>\n")
    # --
    vtufile.write("<DataArray type='Float32' Name='Density' Format='binary'> \n")
    swarm_rho[swarm_active].tofile(vtufile, sep=" ", format="%.3e")
    vtufile.write("</DataArray>\n")
    # --
    vtufile.write("<DataArray type='Float32' Name='id' Format='binary'> \n")
    swarm_id[swarm_active].tofile(vtufile, sep=" ", format="%d")
    vtufile.write("</DataArray>\n")
    # --
    if use_melting:
        vtufile.write("<DataArray type='Float32' Name='F' Format='binary'> \n")
        swarm_F[swarm_active].tofile(vtufile, sep=" ", format="%.3e")
        vtufile.write("</DataArray>\n")
        # --
        vtufile.write("<DataArray type='Float32' Name='super solidus T' Format='binary'> \n")
        swarm_sst[swarm_active].tofile(vtufile, sep=" ", format="%.3e")
        vtufile.write("</DataArray>\n")
    # --
    if solve_Stokes:
        vtufile.write("<DataArray type='Float32' Name='Viscosity' Format='binary'> \n")
        swarm_eta[swarm_active].tofile(vtufile, sep=" ", format="%.3e")
        vtufile.write("</DataArray>\n")
    # --
    vtufile.write("<DataArray type='Float32' Name='Strain' Format='binary'> \n")
    swarm_strain[swarm_active].tofile(vtufile, sep=" ", format="%.3e")
    vtufile.write("</DataArray>\n")
    # --
    if debug_swarm and (geometry == "quarter" or geometry == "half" or geometry == "annulus"):
        vtufile.write("<DataArray type='Float32' Name='rad' Format='binary'> \n")
        swarm_rad[swarm_active].tofile(vtufile, sep=" ", format="%.3e")
        vtufile.write("</DataArray>\n")
        # --
        vtufile.write("<DataArray type='Float32' Name='theta' Format='binary'> \n")
        swarm_theta[swarm_active].tofile(vtufile, sep=" ", format="%.3e")
        vtufile.write("</DataArray>\n")
    # --
    if debug_swarm and solve_T:
        swarm_TK = swarm_T - TKelvin
        vtufile.write("<DataArray type='Float32' Name='Temperature (C)' Format='binary'> \n")
        swarm_TK[swarm_active].tofile(vtufile, sep=" ", format="%.3e")
        vtufile.write("</DataArray>\n")
        # --
        vtufile.write("<DataArray type='Float32' Name='hcond' Format='binary'> \n")
        swarm_hcond[swarm_active].tofile(vtufile, sep=" ", format="%.3e")
        vtufile.write("</DataArray>\n")
        # --
        vtufile.write("<DataArray type='Float32' Name='hcapa' Format='binary'> \n")
        swarm_hcapa[swarm_active].tofile(vtufile, sep=" ", format="%.3e")
        vtufile.write("</DataArray>\n")
        # --
        vtufile.write("<DataArray type='Float32' Name='alpha' Format='binary'> \n")
        swarm_alpha[swarm_active].tofile(vtufile, sep=" ", format="%.3e")
        vtufile.write("</DataArray>\n")
    # --
    vtufile.write("<DataArray type='Int32' Name='Paint' Format='binary'> \n")
    swarm_paint[swarm_active].tofile(vtufile, sep=" ", format="%d")
    vtufile.write("</DataArray>\n")
    # --
    vtufile.write("<DataArray type='Int32' Name='Mechanism' Format='binary'> \n")
    swarm_mechanism[swarm_active].tofile(vtufile, sep=" ", format="%d")
    vtufile.write("</DataArray>\n")
    # --
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    # --
    array = np.arange(0, nparticle_active + 1, dtype=np.int32)
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    array.tofile(vtufile, sep=" ", format="%d")
    vtufile.write("</DataArray>\n")
    # --
    array = np.arange(1, nparticle_active + 2, dtype=np.int32)
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    array.tofile(vtufile, sep=" ", format="%d")
    vtufile.write("</DataArray>\n")
    # --
    array = np.full(nparticle_active, 1, dtype=np.int32)
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    array.tofile(vtufile, sep=" ", format="%d")
    vtufile.write("</DataArray>\n")
    # --
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()


###################################################################################################
