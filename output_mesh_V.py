###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################


def output_mesh_V(x_V,z_V,icon_V,nn_V,m_V,nel):
    """
    """

    filename = "DEBUG/mesh_V.vtu"
    vtufile = open(filename, "w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" % (nn_V, nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0, nn_V):
        vtufile.write("%.5e %.1e %.5e \n" % (x_V[i], 0.0, z_V[i]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<Cells>\n")
    # --
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range(0, nel):
        vtufile.write(
            "%d %d %d %d %d %d %d %d %d\n"
            % (
                icon_V[0, iel],
                icon_V[1, iel],
                icon_V[2, iel],
                icon_V[3, iel],
                icon_V[4, iel],
                icon_V[5, iel],
                icon_V[6, iel],
                icon_V[7, iel],
                icon_V[8, iel],
            )
        )
    vtufile.write("</DataArray>\n")
    # --
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range(0, nel):
        vtufile.write("%d \n" % ((iel + 1) * m_V))
    vtufile.write("</DataArray>\n")
    # --
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range(0, nel):
        vtufile.write("%d \n" % 28)
    vtufile.write("</DataArray>\n")
    # --
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()


###################################################################################################
