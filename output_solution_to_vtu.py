###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np
from toolbox import *

###################################################################################################

def output_solution_to_vtu(solve_Stokes,istep,nel,nn_V,m_V,solve_T,vel_scale,vel_unit,TKelvin,\
                           x_V,z_V,u,w,q,T,eta_nodal,rho_nodal,exx_nodal,ezz_nodal,exz_nodal,\
                           e_nodal,divv_nodal,qx_nodal,qz_nodal,rho_elemental,exx_e,ezz_e,exz_e,\
                           divv_e,sigmaxx_nodal,sigmazz_nodal,sigmaxz_nodal,rad_V,theta_V,
                           eta_elemental,nparticle_elemental,area,icon_V,bc_fix_V,bc_fix_T,geometry,
                           gx_nodal,gz_nodal,err_nodal,ett_nodal,ert_nodal,vr,vt,plith,
                           exx_el,ezz_el,exz_el,taurr_nodal,tautt_nodal,taurt_nodal,
                           particle_rho_projection,particle_eta_projection,ls_rho_a,ls_eta_a):

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
           vtufile.write("%.5e %.1e %.5e \n" %(x_V[i],0.,z_V[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Velocity ("+vel_unit+")' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.4e %.1e %.4e \n" %(u[i]/vel_scale,0.,w[i]/vel_scale))
       vtufile.write("</DataArray>\n")
       #--
       #if debug_sol:
       #   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Normal vector' Format='ascii'> \n")
       #   for i in range(0,nn_V):
       #       vtufile.write("%.3e %.3e %.1e \n" %(nx[i],ny[i],0.))
       #   vtufile.write("</DataArray>\n")
       #--
       if geometry=='quarter' or geometry=='half' or geometry=='eighth': 
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Velocity (Polar)' Format='ascii'> \n")
          for i in range(0,nn_V):
              vtufile.write("%.3e %.1e %.3e \n" %(vr[i]/vel_scale,0.,vt[i]/vel_scale))
          vtufile.write("</DataArray>\n")
       #--
       if solve_Stokes:
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
       if debug_sol and solve_Stokes:
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Gravity vector' Format='ascii'> \n")
          for i in range(0,nn_V):
              vtufile.write("%.5e %.1e %.5e \n" %(gx_nodal[i],0.,gz_nodal[i]))
          vtufile.write("</DataArray>\n")
       #--
       if debug_sol and solve_Stokes:
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
       if solve_Stokes and (not (geometry=='quarter' or geometry=='half' or geometry=='eighth')): 
          vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
          exx_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='ezz' Format='ascii'> \n")
          ezz_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='exz' Format='ascii'> \n")
          exz_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #
          #ee_n=effective(exx,ezz,exz)
          #vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
          #ee_n.tofile(vtufile,sep=' ',format='%.4e')
          #vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='div(v)' Format='ascii'> \n")
          divv_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='sigmaxx' Format='ascii'> \n")
          sigmaxx_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='sigmazz' Format='ascii'> \n")
          sigmazz_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='sigmaxz' Format='ascii'> \n")
          sigmaxz_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
       #--
       if solve_Stokes:
          vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
          e_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
       #--
       if solve_Stokes and (geometry=='quarter' or geometry=='half'): 
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
          ee_p=effective(err_nodal,ett_nodal,ert_nodal)
          vtufile.write("<DataArray type='Float32' Name='e_eff' Format='ascii'> \n")
          ee_p.tofile(vtufile,sep=' ',format='%.4e')
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
       if solve_Stokes:
          if particle_eta_projection=='nodal':
             vtufile.write("<DataArray type='Float32' Name='Viscosity (*)' Format='ascii'> \n")
          else:
             vtufile.write("<DataArray type='Float32' Name='Viscosity' Format='ascii'> \n")
          eta_nodal.tofile(vtufile,sep=' ',format='%.4e')
          vtufile.write("</DataArray>\n")
       #--
       if particle_rho_projection=='nodal':
          vtufile.write("<DataArray type='Float32' Name='Density(*)' Format='ascii'> \n")
       else:
          vtufile.write("<DataArray type='Float32' Name='Density' Format='ascii'> \n")
       rho_nodal.tofile(vtufile,sep=' ',format='%.5e')
       vtufile.write("</DataArray>\n")
       #--
       if solve_T:
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Heat flux' Format='ascii'> \n")
          for i in range(0,nn_V):
              vtufile.write("%.3e %.1e %.3e \n" %(qx_nodal[i],0.,qz_nodal[i]))
          vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       if solve_Stokes:
          vtufile.write("<DataArray type='Float32' Name='div(v)' Format='ascii'> \n")
          divv_e.tofile(vtufile,sep=' ',format='%.5e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
          exx_e.tofile(vtufile,sep=' ',format='%.5e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='ezz' Format='ascii'> \n")
          ezz_e.tofile(vtufile,sep=' ',format='%.5e')
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='exz' Format='ascii'> \n")
          exz_e.tofile(vtufile,sep=' ',format='%.5e')
          vtufile.write("</DataArray>\n")
          #--
       #--
       if solve_Stokes:
          if particle_rho_projection=='elemental':
             vtufile.write("<DataArray type='Float32' Name='Viscosity (*)' Format='ascii'> \n")
          else:
             vtufile.write("<DataArray type='Float32' Name='Viscosity' Format='ascii'> \n")
          for iel in range (0,nel):
              vtufile.write("%.3e\n" % (eta_elemental[iel]))
          vtufile.write("</DataArray>\n")
       #--
       if particle_rho_projection=='elemental':
          vtufile.write("<DataArray type='Float32' Name='Density (*)' Format='ascii'> \n")
       else:
          vtufile.write("<DataArray type='Float32' Name='Density' Format='ascii'> \n")
       rho_elemental.tofile(vtufile,sep=' ',format='%.5e')
       vtufile.write("</DataArray>\n")
       #--
       if particle_rho_projection=='least_squares':
          vtufile.write("<DataArray type='Float32' Name='Density (*)' Format='ascii'> \n")
          ls_rho_a.tofile(vtufile,sep=' ',format='%.5e')
          vtufile.write("</DataArray>\n")
       #--
       if particle_eta_projection=='least_squares':
          vtufile.write("<DataArray type='Float32' Name='Viscosity (*)' Format='ascii'> \n")
          ls_eta_a.tofile(vtufile,sep=' ',format='%.5e')
          vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='nb particles' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" % (nparticle_elemental[iel]))
       vtufile.write("</DataArray>\n")
       #--
       if debug_sol and solve_Stokes and (not (geometry=='quarter' or geometry=='half')): 
          vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
          for iel in range (0,nel):
              vtufile.write("%e \n" % (exx_el[iel]))
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='ezz' Format='ascii'> \n")
          for iel in range (0,nel):
              vtufile.write("%e \n" % (ezz_el[iel]))
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='exz' Format='ascii'> \n")
          for iel in range (0,nel):
              vtufile.write("%e \n" % (exz_el[iel]))
          vtufile.write("</DataArray>\n")
          #
          ee_el=effective(exx_el,ezz_el,exz_el)
          vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
          ee_el.tofile(vtufile,sep=' ',format='%.4e')
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

###################################################################################################
