import numpy as np

###############################################################################

def export_solution_to_vtu(istep,nel,nn_V,m_V,solve_T,vel_scale,TKelvin,x_V,y_V,u,v,q,T,
                           eta_nodal,rho_nodal,exx_nodal,eyy_nodal,exy_nodal,e_nodal,qx_nodal,qy_nodal,
                           rho_elemental,sigmaxx_nodal,sigmayy_nodal,sigmaxy_nodal,rad_V,theta_V,
                           eta_elemental,nparticle_elemental,area,icon_V,bc_fix_V,bc_fix_T,geometry,
                           gx_nodal,gy_nodal,err_nodal,ett_nodal,ert_nodal,vr,vt,plith,nx,ny,
                           exx_el,eyy_el,exy_el,taurr_nodal,tautt_nodal,taurt_nodal):

       debug_sol=False

       filename = 'OUTPUT/solution_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.5e %.5e %.1e \n" %(x_V[i],y_V[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Velocity' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.4e %.4e %.1e \n" %(u[i]/vel_scale,v[i]/vel_scale,0.))
       vtufile.write("</DataArray>\n")
       #--
       if debug_sol:
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Normal vector' Format='ascii'> \n")
          for i in range(0,nn_V):
              vtufile.write("%.3e %.3e %.1e \n" %(nx[i],ny[i],0.))
          vtufile.write("</DataArray>\n")
       #--
       if geometry=='quarter' or geometry=='half': 
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Velocity (Polar)' Format='ascii'> \n")
          for i in range(0,nn_V):
              vtufile.write("%.3e %.3e %.1e \n" %(vr[i]/vel_scale,vt[i]/vel_scale,0.))
          vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Pressure' Format='ascii'> \n")
       q.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Pressure (lith)' Format='ascii'> \n")
       plith.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Pressure (dyn)' Format='ascii'> \n")
       (q-plith).tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")


       #--
       if solve_T:
          vtufile.write("<DataArray type='Float32' Name='Temperature' Format='ascii'> \n")
          for i in range(0,nn_V):
              vtufile.write("%.4e \n" %(T[i]-TKelvin))
          vtufile.write("</DataArray>\n")
       #--
       if debug_sol:
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Gravity vector' Format='ascii'> \n")
          for i in range(0,nn_V):
              vtufile.write("%.3e %.3e %.1e \n" %(gx_nodal[i],gy_nodal[i],0.))
          vtufile.write("</DataArray>\n")
       #--
       if debug_sol:
          vtufile.write("<DataArray type='Int32' Name='fix bc u' Format='ascii'> \n")
          for i in range(0,nn_V):
              if bc_fix_V[2*i]: 
                 val=1
              else:
                 val=0 
              vtufile.write("%d \n" % val)
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Int32' Name='fix bc v' Format='ascii'> \n")
          for i in range(0,nn_V):
              if bc_fix_V[2*i+1]: 
                 val=1
              else:
                 val=0 
              vtufile.write("%d \n" % val)
          vtufile.write("</DataArray>\n")
       #--
       if debug_sol and solve_T:
          vtufile.write("<DataArray type='Int32' Name='fix bc T' Format='ascii'> \n")
          for i in range(0,nn_V):
              if bc_fix_T[i]: 
                 val=1
              else:
                 val=0 
              vtufile.write("%d \n" % val)
          vtufile.write("</DataArray>\n")

       #--
       if not (geometry=='quarter' or geometry=='half'): 
          vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
          exx_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
          eyy_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
          exy_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='sigmaxx' Format='ascii'> \n")
          sigmaxx_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='sigmayy' Format='ascii'> \n")
          sigmayy_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='sigmaxy' Format='ascii'> \n")
          sigmaxy_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")

       #--
       vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
       e_nodal.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       if geometry=='quarter' or geometry=='half': 
          vtufile.write("<DataArray type='Float32' Name='err' Format='ascii'> \n")
          err_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='ett' Format='ascii'> \n")
          ett_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='ert' Format='ascii'> \n")
          ert_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='taurr' Format='ascii'> \n")
          taurr_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='tautt' Format='ascii'> \n")
          tautt_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='taurt' Format='ascii'> \n")
          taurt_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")

       #--
       if debug_sol and (geometry=='quarter' or geometry=='half'):  
          vtufile.write("<DataArray type='Float32' Name='rad' Format='ascii'> \n")
          rad_V.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='theta' Format='ascii'> \n")
          theta_V.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--






       #--
       vtufile.write("<DataArray type='Float32' Name='Viscosity' Format='ascii'> \n")
       eta_nodal.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Density' Format='ascii'> \n")
       rho_nodal.tofile(vtufile,sep=' ',format='%.4e')
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
           vtufile.write("%.4e\n" % (rho_elemental[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='nb particles' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" % (nparticle_elemental[iel]))
       vtufile.write("</DataArray>\n")
       #--
       if debug_sol:
          vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
          for iel in range (0,nel):
              vtufile.write("%e \n" % (exx_el[iel]))
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
          for iel in range (0,nel):
              vtufile.write("%e \n" % (eyy_el[iel]))
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
          for iel in range (0,nel):
              vtufile.write("%e \n" % (exy_el[iel]))
          vtufile.write("</DataArray>\n")

       #--
       if debug_sol:
          vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
          for iel in range (0,nel):
              vtufile.write("%e \n" % (area[iel]))
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
