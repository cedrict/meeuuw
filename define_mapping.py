import numpy as np

###############################################################################
# Q1 nodes for mapping are corners of Q2 basis functions
# Q2 nodes for mapping are same as Q2 basis functions
# Q4 nodes for mapping are built
# note that python uses row-major storage for 2D arrays and since we need 
# to do dot products with x_M and z_M it makes more sense 
# to use column-major, i.e. F_CONTIGUOUS in python jargon.
###############################################################################

def define_mapping(geometry,mapping,nelx,nel,x_V,z_V,icon_V,rad_V,theta_V):

   if mapping=='Q1': m_M=4
   if mapping=='Q2': m_M=9
   if mapping=='Q4': m_M=25

   x_M=np.zeros((m_M,nel),dtype=np.float64,order='F')
   z_M=np.zeros((m_M,nel),dtype=np.float64,order='F')

   match mapping:
    case 'Q1':
     for iel in range(0,nel):
         x_M[0,iel]=x_V[icon_V[0,iel]] ; z_M[0,iel]=z_V[icon_V[0,iel]]
         x_M[1,iel]=x_V[icon_V[1,iel]] ; z_M[1,iel]=z_V[icon_V[1,iel]]
         x_M[2,iel]=x_V[icon_V[2,iel]] ; z_M[2,iel]=z_V[icon_V[2,iel]]
         x_M[3,iel]=x_V[icon_V[3,iel]] ; z_M[3,iel]=z_V[icon_V[3,iel]]

    case 'Q2':
     for iel in range(0,nel):
         x_M[0,iel]=x_V[icon_V[0,iel]] ; z_M[0,iel]=z_V[icon_V[0,iel]]
         x_M[1,iel]=x_V[icon_V[1,iel]] ; z_M[1,iel]=z_V[icon_V[1,iel]]
         x_M[2,iel]=x_V[icon_V[2,iel]] ; z_M[2,iel]=z_V[icon_V[2,iel]]
         x_M[3,iel]=x_V[icon_V[3,iel]] ; z_M[3,iel]=z_V[icon_V[3,iel]]
         x_M[4,iel]=x_V[icon_V[4,iel]] ; z_M[4,iel]=z_V[icon_V[4,iel]]
         x_M[5,iel]=x_V[icon_V[5,iel]] ; z_M[5,iel]=z_V[icon_V[5,iel]]
         x_M[6,iel]=x_V[icon_V[6,iel]] ; z_M[6,iel]=z_V[icon_V[6,iel]]
         x_M[7,iel]=x_V[icon_V[7,iel]] ; z_M[7,iel]=z_V[icon_V[7,iel]]
         x_M[8,iel]=x_V[icon_V[8,iel]] ; z_M[8,iel]=z_V[icon_V[8,iel]]

    case 'Q4':

     dtheta=np.pi/2/(2*nelx)
     for iel in range(0,nel):
            thetamin=np.pi/2-theta_V[icon_V[0,iel]]
            rmin=rad_V[icon_V[0,iel]]
            rmax=rad_V[icon_V[2,iel]]
            counter=0
            for j in range(0,5):
                for i in range(0,5):
                    ttt=thetamin+i*dtheta
                    rrr=rmin+j*(rmax-rmin)/4
                    x_M[counter,iel]=np.sin(ttt)*rrr
                    z_M[counter,iel]=np.cos(ttt)*rrr
                    counter+=1

    case _ :

     exit("define_mapping: unknown mapping")

   return x_M,z_M,m_M

###############################################################################
