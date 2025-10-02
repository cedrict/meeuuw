import numpy as np
import numba

## PB: shear heating should use dev strain rate

###############################################################################

@numba.njit
def build_matrix_energy(bignb,nel,nqel,m_T,Nfem_T,T,icon_V,rhoq,etaq,Tq,uq,vq,\
                        hcondq,hcapaq,exxq,eyyq,exyq,dpdxq,dpdyq,JxWq,N_V,dNdx_V,dNdy_V,\
                        bc_fix_T,bc_val_T,dt,formulation,rho0):

    VV_T=np.zeros(bignb,dtype=np.float64)    

    Tvect=np.zeros(m_T,dtype=np.float64)   
    rhs=np.zeros(Nfem_T,dtype=np.float64)    # FE rhs 
    B=np.zeros((2,m_T),dtype=np.float64)     # gradient matrix B 

    counter=0
    for iel in range(0,nel):

        b_el=np.zeros(m_T,dtype=np.float64)
        A_el=np.zeros((m_T,m_T),dtype=np.float64)
        Ka=np.zeros((m_T,m_T),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m_T,m_T),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m_T,m_T),dtype=np.float64)   # elemental mass matrix 
        velq=np.zeros((1,2),dtype=np.float64)

        Tvect[0:m_T]=T[icon_V[0:m_T,iel]]

        for iq in range(0,nqel):

            N=N_V[iq,:]

            velq[0,0]=uq[iel,iq]
            velq[0,1]=vq[iel,iq]

            B[0,:]=dNdx_V[iq,:]
            B[1,:]=dNdy_V[iq,:]
            
            MM+=np.outer(N,N)*rho0*hcapaq[iel,iq]*JxWq[iq] # mass matrix

            Kd+=B.T.dot(B)*hcondq[iel,iq]*JxWq[iq] # diffusion matrix
            
            Ka+=np.outer(N,velq.dot(B))*rho0*hcapaq[iel,iq]*JxWq[iq] # advection matrix

            #if formulation=='EBA':
               #viscous dissipation
               #b_el[:]+=N[:]*JxWq[iq]*2*etaq[iel,iq]*\
               #         (exxq[iel,iq]**2+eyyq[iel,iq]**2+2*exyq[iel,iq]**2) 
               #adiabatic heating
               #b_el[:]+=N[:]*JxWq[iq]*alphaT*Tq*(velq[0,0]*dpdxq[iel,iq]+velq[0,1]*dpdyq[iel,iq])  

        #end for

        A_el+=MM+(Ka+Kd)*dt*0.5
        b_el+=(MM-(Ka+Kd)*dt*0.5).dot(Tvect)

        # apply boundary conditions
        for k1 in range(0,m_T):
            m1=icon_V[k1,iel]
            if bc_fix_T[m1]:
               Aref=A_el[k1,k1]
               for k2 in range(0,m_T):
                   m2=icon_V[k2,iel]
                   b_el[k2]-=A_el[k2,k1]*bc_val_T[m1]
                   A_el[k1,k2]=0
                   A_el[k2,k1]=0
               #end for
               A_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val_T[m1]
            #end for
        #end for

        # assemble matrix K_mat and right hand side rhs
        for ikk in range(m_T):
            m1=icon_V[ikk,iel]
            for jkk in range(m_T):
                VV_T[counter]=A_el[ikk,jkk]
                counter+=1
            rhs[m1]+=b_el[ikk]
        #end for


    return VV_T,rhs

###############################################################################
