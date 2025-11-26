import numpy as np

def compute_pressure_average(geometry,pressure_normalisation,axisymmetric,top_element,\
                             nel,nelx,nqel,N_P,JxWq,p,icon_P,theta_P,xq,volume):

    match(pressure_normalisation): 

     case('surface'):

         if axisymmetric:
            pressure_average=0.
            for iel in range(0,nel):
                if top_element[iel]:
                   pmean=0.5*(p[icon_P[2,iel]]+p[icon_P[3,iel]])
                   theta_mean=np.pi/2-0.5*(theta_P[icon_P[2,iel]]+theta_P[icon_P[3,iel]])
                   dtheta=theta_P[icon_P[3,iel]]-theta_P[icon_P[2,iel]]
                   pressure_average+=0.5*pmean*np.sin(theta_mean)*dtheta
         else:
            pressure_average=0.
            for iel in range(0,nel):
                if top_element[iel]:
                   pressure_average+=0.5*(p[icon_P[2,iel]]+p[icon_P[3,iel]])
            pressure_average/=nelx

     case('volume'):

         if axisymmetric:
            pressure_average=0.
            for iel in range(0,nel):
                for iq in range(0,nqel):
                    pressure_average+=np.dot(N_P[iq,:],p[icon_P[:,iel]])*JxWq[iel,iq]*2*np.pi*xq[iel,iq]
         else:
            pressure_average=0
            for iel in range(0,nel):
                for iq in range(0,nqel):
                    pressure_average+=np.dot(N_P[iq,:],p[icon_P[:,iel]])*JxWq[iel,iq]
            pressure_average/=volume

     case _ :
      exit('pressure_normalisation: unknown value')

    return pressure_average
