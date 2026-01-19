###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################
import numpy as np

###################################################################################################

def output_swarm_to_vtu(solve_Stokes,use_melting,TKelvin,istep,geometry,nparticle,solve_T,\
                        vel_scale,swarm_x,swarm_z,swarm_u,swarm_w,swarm_mat,swarm_rho,swarm_eta,\
                        swarm_r,swarm_t,swarm_p,swarm_paint,swarm_exx,swarm_ezz,swarm_exz,swarm_T,\
                        swarm_iel,swarm_hcond,swarm_hcapa,swarm_rad,swarm_theta,swarm_strain,\
                        swarm_F,swarm_sst):

       debug_swarm=False

       filename='OUTPUT/swarm_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nparticle,nparticle))
       #####
       vtufile.write("<Points> \n")
       #--
       swarm_X=np.zeros(nparticle,dtype=np.float64) 
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       coords=np.array([swarm_x,swarm_X,swarm_z]).T
       coords.tofile(vtufile,sep=' ')

       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       if debug_swarm and solve_Stokes:
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Velocity' Format='ascii'> \n")
          coords=np.array([swarm_u,swarm_X,swarm_w]).T
          coords.tofile(vtufile,sep=' ')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='exx' Format='binary'> \n")
          swarm_exx.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='ezz' Format='binary'> \n")
          swarm_ezz.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='exz' Format='binary'> \n")
          swarm_exz.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='Pressure' Format='binary'> \n")
          swarm_p.tofile(vtufile,sep=' ',format='%.5e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='r' Format='binary'> \n")
          swarm_r.tofile(vtufile,sep=' ',format='%.3e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='t' Format='binary'> \n")
          swarm_t.tofile(vtufile,sep=' ',format='%.3e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Int32' Name='iel' Format='binary'> \n")
          swarm_iel.tofile(vtufile,sep=' ')
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
       if use_melting:
          vtufile.write("<DataArray type='Float32' Name='F' Format='binary'> \n")
          swarm_F.tofile(vtufile,sep=' ',format='%.3e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='super solidus T' Format='binary'> \n")
          swarm_sst.tofile(vtufile,sep=' ',format='%.3e')
          vtufile.write("</DataArray>\n")
       #--
       if solve_Stokes:
          vtufile.write("<DataArray type='Float32' Name='Viscosity' Format='binary'> \n")
          swarm_eta.tofile(vtufile,sep=' ',format='%.3e')
          vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Strain' Format='binary'> \n")
       swarm_strain.tofile(vtufile,sep=' ',format='%.3e')
       vtufile.write("</DataArray>\n")
       #--
       if debug_swarm and geometry=='quarter':
          vtufile.write("<DataArray type='Float32' Name='rad' Format='binary'> \n")
          swarm_rad.tofile(vtufile,sep=' ',format='%.3e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='theta' Format='binary'> \n")
          swarm_theta.tofile(vtufile,sep=' ',format='%.3e')
          vtufile.write("</DataArray>\n")
       #--
       if debug_swarm and solve_T:
          swarm_TK=swarm_T-TKelvin
          vtufile.write("<DataArray type='Float32' Name='Temperature' Format='binary'> \n")
          swarm_TK.tofile(vtufile,sep=' ',format='%.3e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='hcond' Format='binary'> \n")
          swarm_hcond.tofile(vtufile,sep=' ',format='%.3e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='hcapa' Format='binary'> \n")
          swarm_hcapa.tofile(vtufile,sep=' ',format='%.3e')
          vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='Paint' Format='binary'> \n")
       swarm_paint.tofile(vtufile,sep=' ',format='%.3e')
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

###################################################################################################
