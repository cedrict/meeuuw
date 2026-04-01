###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np
import matplotlib.pyplot as plt

###################################################################################################

def output_solution_to_pdf(geometry,solve_Stokes,solve_T,istep,vel_scale,vel_unit,TKelvin,nelx,nelz,\
                           Lx,Lz,x_V,z_V,u,w,q,T,eta_n,rho_n,exx_n,ezz_n,exz_n,e_n,divv_n,qx_n,qz_n):
                           
    if geometry=='box':
       nnx=2*nelx+1
       nnz=2*nelz+1
       vel=np.sqrt(u**2+w**2)
       x2=np.linspace(0,Lx,nnx)
       z2=np.linspace(0,Lz,nnz)
       T2=np.reshape(T,(nnx,nnz))
       u2=np.reshape(u,(nnx,nnz))
       w2=np.reshape(w,(nnx,nnz))
       vel2=np.reshape(vel,(nnx,nnz))
       exx2=np.reshape(exx_n,(nnx,nnz))
       ezz2=np.reshape(ezz_n,(nnx,nnz))
       exz2=np.reshape(exz_n,(nnx,nnz))
       e2=np.reshape(e_n,(nnx,nnz))
       q2=np.reshape(q,(nnx,nnz))
       qx2=np.reshape(qx_n,(nnx,nnz))
       qz2=np.reshape(qz_n,(nnx,nnz))
       eta2=np.reshape(eta_n,(nnx,nnz))
       rho2=np.reshape(rho_n,(nnx,nnz))
       divv2=np.reshape(divv_n,(nnx,nnz))

       plt.figure()
       col=plt.pcolormesh(x2,z2,T2,cmap='coolwarm') ; plt.colorbar(col)
       plt.title("Temperature") ; plt.xlabel("x") ; plt.ylabel("z")
       filename = 'OUTPUT/solution_T_{:04d}.pdf'.format(istep)
       plt.savefig(filename, bbox_inches='tight')

       plt.figure()
       col=plt.pcolormesh(x2,z2,u2,cmap='coolwarm') ; plt.colorbar(col)
       plt.title("Velocity x-component") ; plt.xlabel("x") ; plt.ylabel("z")
       filename = 'OUTPUT/solution_u_{:04d}.pdf'.format(istep)
       plt.savefig(filename, bbox_inches='tight')

       plt.figure()
       col=plt.pcolormesh(x2,z2,w2,cmap='coolwarm') ; plt.colorbar(col)
       plt.title("Velocity z-component") ; plt.xlabel("x") ; plt.ylabel("z")
       filename = 'OUTPUT/solution_w_{:04d}.pdf'.format(istep)
       plt.savefig(filename, bbox_inches='tight')

       plt.figure()
       col=plt.pcolormesh(x2,z2,vel2,cmap='coolwarm') ; plt.colorbar(col)
       plt.title("Velocity norm") ; plt.xlabel("x") ; plt.ylabel("z")
       filename = 'OUTPUT/solution_vel_{:04d}.pdf'.format(istep)
       plt.savefig(filename, bbox_inches='tight')

       plt.figure()
       col=plt.pcolormesh(x2,z2,rho2,cmap='coolwarm') ; plt.colorbar(col)
       plt.title("Density") ; plt.xlabel("x") ; plt.ylabel("z")
       filename = 'OUTPUT/solution_rho_{:04d}.pdf'.format(istep)
       plt.savefig(filename, bbox_inches='tight')

       plt.figure()
       col=plt.pcolormesh(x2,z2,divv2,cmap='coolwarm') ; plt.colorbar(col)
       plt.title("Velocity divergence") ; plt.xlabel("x") ; plt.ylabel("z")
       filename = 'OUTPUT/solution_divv_{:04d}.pdf'.format(istep)
       plt.savefig(filename, bbox_inches='tight')

       plt.figure()
       col=plt.pcolormesh(x2,z2,e2,cmap='coolwarm') ; plt.colorbar(col)
       plt.title("Effective strain rate") ; plt.xlabel("x") ; plt.ylabel("z")
       filename = 'OUTPUT/solution_e_{:04d}.pdf'.format(istep)
       plt.savefig(filename, bbox_inches='tight')

       plt.figure()
       col=plt.pcolormesh(x2,z2,q2,cmap='coolwarm') ; plt.colorbar(col)
       plt.title("Pressure") ; plt.xlabel("x") ; plt.ylabel("z")
       filename = 'OUTPUT/solution_p_{:04d}.pdf'.format(istep)
       plt.savefig(filename, bbox_inches='tight')

       plt.figure()
       col=plt.pcolormesh(x2,z2,eta2,cmap='coolwarm') ; plt.colorbar(col)
       plt.title("Viscosity") ; plt.xlabel("x") ; plt.ylabel("z")
       filename = 'OUTPUT/solution_eta_{:04d}.pdf'.format(istep)
       plt.savefig(filename, bbox_inches='tight')

       plt.figure()
       col=plt.pcolormesh(x2,z2,qx2,cmap='coolwarm') ; plt.colorbar(col)
       plt.title("Heat flux x-component") ; plt.xlabel("x") ; plt.ylabel("z")
       filename = 'OUTPUT/solution_qx_{:04d}.pdf'.format(istep)
       plt.savefig(filename, bbox_inches='tight')

       plt.figure()
       col=plt.pcolormesh(x2,z2,qz2,cmap='coolwarm') ; plt.colorbar(col)
       plt.title("Heat flux z-component") ; plt.xlabel("x") ; plt.ylabel("z")
       filename = 'OUTPUT/solution_qz_{:04d}.pdf'.format(istep)
       plt.savefig(filename, bbox_inches='tight')

       plt.figure()
       col=plt.pcolormesh(x2,z2,exx2,cmap='coolwarm') ; plt.colorbar(col)
       plt.title("Strain rate e_xx") ; plt.xlabel("x") ; plt.ylabel("z")
       filename = 'OUTPUT/solution_exx_{:04d}.pdf'.format(istep)
       plt.savefig(filename, bbox_inches='tight')

       plt.figure()
       col=plt.pcolormesh(x2,z2,ezz2,cmap='coolwarm') ; plt.colorbar(col)
       plt.title("Strain rate e_zz") ; plt.xlabel("x") ; plt.ylabel("z")
       filename = 'OUTPUT/solution_ezz_{:04d}.pdf'.format(istep)
       plt.savefig(filename, bbox_inches='tight')

       plt.figure()
       col=plt.pcolormesh(x2,z2,exz2,cmap='coolwarm') ; plt.colorbar(col)
       plt.title("Strain rate e_xz") ; plt.xlabel("x") ; plt.ylabel("z")
       filename = 'OUTPUT/solution_exz_{:04d}.pdf'.format(istep)
       plt.savefig(filename, bbox_inches='tight')

    else:

       print('     matplotlib export unavailable for this geometry')


###################################################################################################
