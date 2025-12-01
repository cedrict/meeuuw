import numba

###################################################################################################
# this functions exports the values of the solution fields at a given set of user-chosen 
# locations, provided these locations correspond to a V node location.
# A more versatile algorithm could/should be implemented in the future.
###################################################################################################

@numba.njit
def sample_solution_box(nn_V,x_V,z_V,u,w,q,T,nsamplepoints,xsamplepoints,zsamplepoints,Lx,Lz,nelx,nelz):
    """
    Args:
     nn_V: number of V nodes
     x_V,z_V: coordinates of V nodes
     u,w,q,T: fields on V nodes
     nsamplepoints: nb of sampling points
     xsamplepoints,zsamplepoints: coordinates of sampling points
     Lx,Lz: domain size
     nelx,nelz: number of elements
    Returns:
     -
    """

    for isp in range(0,nsamplepoints):
        for i in range(nn_V):
            if abs(x_V[i]-xsamplepoints[isp])/Lx<1e-6 and\
               abs(z_V[i]-zsamplepoints[isp])/Lz<1e-6:
               print('sample ->',x_V[i],z_V[i],u[i],w[i],q[i],T[i],nelx,nelz)

###################################################################################################
