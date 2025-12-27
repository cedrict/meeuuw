###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################

def output_quadpoints_to_vtu(istep,nel,nq_per_element,nq,solve_T,xq,zq,rhoq,etaq,Tq,\
                             hcondq,hcapaq,dpdxq,dpdzq,gxq,gzq):

       filename='OUTPUT/quadpoints_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nq,nq))
       #####
       vtufile.write("<Points> \n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for iel in range(nel):
           for iq in range(0,nq_per_element):
               vtufile.write("%.4e %.1e %.4e \n" %(xq[iel,iq],0.,zq[iel,iq]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Density' Format='binary'> \n")
       for iel in range(nel):
           for iq in range(0,nq_per_element):
               vtufile.write("%.5e \n" %(rhoq[iel,iq]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Viscosity' Format='binary'> \n")
       for iel in range(nel):
           for iq in range(0,nq_per_element):
               vtufile.write("%.5e \n" %(etaq[iel,iq]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Pressure gradient' Format='binary'> \n")
       for iel in range(nel):
           for iq in range(0,nq_per_element):
               vtufile.write("%.3e %.1e %.3e \n" %(dpdxq[iel,iq],0.,dpdzq[iel,iq]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Gravity vector' Format='binary'> \n")
       for iel in range(nel):
           for iq in range(0,nq_per_element):
               vtufile.write("%.3e %.1e %.3e \n" %(gxq[iel,iq],0.,gzq[iel,iq]))
       vtufile.write("</DataArray>\n")
       #--
       if solve_T:
          vtufile.write("<DataArray type='Float32' Name='Temperature' Format='binary'> \n")
          for iel in range(nel):
              for iq in range(0,nq_per_element):
                  vtufile.write("%.5e \n" %(Tq[iel,iq]))
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='Heat conductivity' Format='binary'> \n")
          for iel in range(nel):
              for iq in range(0,nq_per_element):
                  vtufile.write("%.5e \n" %(hcondq[iel,iq]))
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='Heat capacity' Format='binary'> \n")
          for iel in range(nel):
              for iq in range(0,nq_per_element):
                  vtufile.write("%.5e \n" %(hcapaq[iel,iq]))
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

###################################################################################################
