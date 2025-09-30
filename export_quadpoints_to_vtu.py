import numpy as np

###############################################################################

def export_quadpoints_to_vtu(istep,nel,nqel,nq,xq,yq,rhoq,etaq):

       filename='quadpoints_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nq,nq))
       #####
       vtufile.write("<Points> \n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for iel in range(nel):
           for iq in range(0,nqel):
               vtufile.write("%.3e %.3e %.1e \n" %(xq[iel,iq],yq[iel,iq],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Density' Format='binary'> \n")
       for iel in range(nel):
           for iq in range(0,nqel):
               vtufile.write("%.5e \n" %(rhoq[iel,iq]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Viscosity' Format='binary'> \n")
       for iel in range(nel):
           for iq in range(0,nqel):
               vtufile.write("%.5e \n" %(etaq[iel,iq]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iq in range (0,nq):
           vtufile.write("%d " % iq)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iq in range (0,nq):
           vtufile.write("%d " % (iq+1) )
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iq in range (0,nq):
           vtufile.write("%d " % 1)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

###############################################################################
