import numpy as np

###############################################################################

def export_solution_to_vtu(istep,nel,nn_V,m_V,solve_T,vel_scale,TKelvin,x_V,y_V,u,v,q,T,
                           eta_nodal,rho_nodal,exx_nodal,eyy_nodal,exy_nodal,qx_nodal,qy_nodal,
                           rho_elemental,eta_elemental,nparticle_elemental,icon_V):


       filename = 'solution_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.3e %.3e %.1e \n" %(x_V[i],y_V[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Velocity' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.3e %.3e %.1e \n" %(u[i]/vel_scale,v[i]/vel_scale,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Pressure' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.3e \n" % (q[i]))
       vtufile.write("</DataArray>\n")
       #--
       if solve_T:
          vtufile.write("<DataArray type='Float32' Name='Temperature' Format='ascii'> \n")
          for i in range(0,nn_V):
              vtufile.write("%.3e \n" %(T[i]-TKelvin))
          vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.4e \n" %exx_nodal[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.4e \n" %eyy_nodal[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.4e \n" %exy_nodal[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Viscosity' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.3e \n" %eta_nodal[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Density' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.5e \n" %rho_nodal[i])
       vtufile.write("</DataArray>\n")
       #--
       if solve_T:
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Heat flux' Format='ascii'> \n")
          for i in range(0,nn_V):
              vtufile.write("%.3e %.3e %.1e \n" %(qx_nodal[i],qy_nodal[i],0.))
          vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Viscosity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%.3e\n" % (eta_elemental[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Density' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%.3e\n" % (rho_elemental[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='nb particles' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" % (nparticle_elemental[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],\
                                                          icon_V[3,iel],icon_V[4,iel],icon_V[5,iel],\
                                                          icon_V[6,iel],icon_V[7,iel],icon_V[8,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*m_V))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %28)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

###############################################################################
