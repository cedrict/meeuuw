import numpy as np

###############################################################################

def export_swarm_to_vtu(istep,nparticle,solve_T,vel_scale,swarm_x,swarm_y,\
                        swarm_u,swarm_v,swarm_mat,swarm_rho,swarm_eta,\
                        swarm_paint,swarm_exx,swarm_eyy,swarm_exy,swarm_T,\
                        swarm_hcond,swarm_hcapa):

       debug_swarm=False

       filename='swarm_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nparticle,nparticle))
       #####
       vtufile.write("<Points> \n")
       #--
       swarm_z=np.zeros(nparticle,dtype=np.float64) 
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       coords=np.array([swarm_x,swarm_y,swarm_z]).T
       coords.tofile(vtufile,sep=' ')

       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       if debug_swarm:
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
          coords=np.array([swarm_u,swarm_v,swarm_z]).T
          coords.tofile(vtufile,sep=' ')
          vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='mat' Format='binary'> \n")
       swarm_mat.tofile(vtufile,sep=' ')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Density' Format='binary'> \n")
       swarm_rho.tofile(vtufile,sep=' ')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Viscosity' Format='binary'> \n")
       swarm_eta.tofile(vtufile,sep=' ')
       vtufile.write("</DataArray>\n")
       #--
       if debug_swarm:
          vtufile.write("<DataArray type='Float32' Name='exx' Format='binary'> \n")
          swarm_exx.tofile(vtufile,sep=' ')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='eyy' Format='binary'> \n")
          swarm_eyy.tofile(vtufile,sep=' ')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='exy' Format='binary'> \n")
          swarm_exy.tofile(vtufile,sep=' ')
          vtufile.write("</DataArray>\n")
       #--
       if debug_swarm and solve_T:
          vtufile.write("<DataArray type='Float32' Name='Temperature' Format='binary'> \n")
          swarm_T.tofile(vtufile,sep=' ')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='hcond' Format='binary'> \n")
          swarm_hcond.tofile(vtufile,sep=' ')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='hcapa' Format='binary'> \n")
          swarm_hcapa.tofile(vtufile,sep=' ')
          vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='Paint' Format='binary'> \n")
       swarm_paint.tofile(vtufile,sep=' ')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       array=np.arange(0,nparticle+1,dtype=np.int32)
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       array.tofile(vtufile,sep=' ')
       vtufile.write("</DataArray>\n")
       #--
       array=np.arange(1,nparticle+2,dtype=np.int32)
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       array.tofile(vtufile,sep=' ')
       vtufile.write("</DataArray>\n")
       #--
       array=np.full(nparticle,1,dtype=np.int32)
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       array.tofile(vtufile,sep=' ')
       vtufile.write("</DataArray>\n")
    
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

###############################################################################
