###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

from vtu_binary import VTKBinaryAppendedWriter, write_text

###################################################################################################


def output_quadpoints_to_vtu(
    istep,
    nel,
    nq_per_element,
    nq,
    solve_T,
    xq,
    zq,
    rhoq,
    etaq,
    Tq,
    hcondq,
    hcapaq,
    dpdxq,
    dpdzq,
    gxq,
    gzq,
    output_folder,
):
    """
    Args:
    Returns:
    """


    filename = output_folder+"/quadpoints_{:04d}.vtu".format(istep)

    writer = VTKBinaryAppendedWriter()

    # -------------------------------------------------------------------------
    # Flatten quadrature point arrays
    # -------------------------------------------------------------------------
    # Old loop order was:
    #
    # for iel in range(nel):
    #     for iq in range(nq_per_element):
    #
    # np.ravel() with default C-order gives the same ordering for arrays shaped
    # like [nel, nq_per_element].

    xq_flat = xq.ravel()
    zq_flat = zq.ravel()

    rhoq_flat = rhoq.ravel()
    etaq_flat = etaq.ravel()

    dpdxq_flat = dpdxq.ravel()
    dpdzq_flat = dpdzq.ravel()

    gxq_flat = gxq.ravel()
    gzq_flat = gzq.ravel()

    # -------------------------------------------------------------------------
    # Points
    # -------------------------------------------------------------------------

    quad_Y = np.zeros(nq, dtype=np.float32)
    coords = np.array([xq_flat, quad_Y, zq_flat]).T

    # -------------------------------------------------------------------------
    # Cells: one VTK_VERTEX per quadrature point
    # -------------------------------------------------------------------------

    connectivity = np.arange(0, nq, dtype=np.int32)
    offsets = np.arange(1, nq + 1, dtype=np.int32)
    types = np.full(nq, 1, dtype=np.uint8)  # VTK_VERTEX = 1

    # -------------------------------------------------------------------------
    # Open binary file
    # -------------------------------------------------------------------------

    with open(filename, "wb") as vtufile:

        def write(text):
            write_text(vtufile, text)

        # ---------------------------------------------------------------------
        # Header
        # ---------------------------------------------------------------------

        write(
            "<VTKFile "
            "type='UnstructuredGrid' "
            "version='0.1' "
            f"byte_order='{writer.byte_order}' "
            f"header_type='{writer.header_type}'>\n"
        )

        write("<UnstructuredGrid>\n")

        write(
            "<Piece "
            "NumberOfPoints='{:d}' "
            "NumberOfCells='{:d}'>\n".format(
                nq,
                nq,
            )
        )

        # ---------------------------------------------------------------------
        # Points
        # ---------------------------------------------------------------------

        write("<Points>\n")
        write(writer.add_points(coords))
        write("</Points>\n")

        # ---------------------------------------------------------------------
        # PointData
        # ---------------------------------------------------------------------

        write("<PointData Scalars='scalars'>\n")

        write(
            writer.add_array(
                name="Density",
                array=rhoq_flat,
                dtype=np.float32,
                vtk_type="Float32",
            )
        )

        write(
            writer.add_array(
                name="Viscosity",
                array=etaq_flat,
                dtype=np.float32,
                vtk_type="Float32",
            )
        )

        pressure_gradient = np.array([dpdxq_flat, quad_Y, dpdzq_flat]).T

        write(
            writer.add_array(
                name="Pressure gradient",
                array=pressure_gradient,
                dtype=np.float32,
                vtk_type="Float32",
                number_of_components=3,
            )
        )

        gravity_vector = np.array([gxq_flat, quad_Y, gzq_flat]).T

        write(
            writer.add_array(
                name="Gravity vector",
                array=gravity_vector,
                dtype=np.float32,
                vtk_type="Float32",
                number_of_components=3,
            )
        )

        if solve_T:

            write(
                writer.add_array(
                    name="Temperature",
                    array=Tq.ravel(),
                    dtype=np.float32,
                    vtk_type="Float32",
                )
            )

            write(
                writer.add_array(
                    name="Heat conductivity",
                    array=hcondq.ravel(),
                    dtype=np.float32,
                    vtk_type="Float32",
                )
            )

            write(
                writer.add_array(
                    name="Heat capacity",
                    array=hcapaq.ravel(),
                    dtype=np.float32,
                    vtk_type="Float32",
                )
            )

        write("</PointData>\n")

        # ---------------------------------------------------------------------
        # Cells
        # ---------------------------------------------------------------------

        write("<Cells>\n")

        write(
            writer.add_array(
                name="connectivity",
                array=connectivity,
                dtype=np.int32,
                vtk_type="Int32",
            )
        )

        write(
            writer.add_array(
                name="offsets",
                array=offsets,
                dtype=np.int32,
                vtk_type="Int32",
            )
        )

        write(
            writer.add_array(
                name="types",
                array=types,
                dtype=np.uint8,
                vtk_type="UInt8",
            )
        )

        write("</Cells>\n")

        # ---------------------------------------------------------------------
        # Close XML and write appended binary data
        # ---------------------------------------------------------------------

        write("</Piece>\n")
        write("</UnstructuredGrid>\n")

        writer.write_appended_data(vtufile)

        write("</VTKFile>\n")


###################################################################################################
