import numpy as np
import numba

###############################################################################

@numba.njit
def build_matrix_stokes(bignb,nel,nqel,m_V,m_P,ndof_V,Nfem_V,Nfem,ndof_V_el,icon_V,icon_P,\
                        rhoq,etaq,JxWq,local_to_globalV,gy,Ly,N_V,N_P,dNdx_V,dNdy_V,\
                        eta_ref,L_ref,bc_fix_V,bc_val_V):

    C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

    VV_V=np.zeros(bignb,dtype=np.float64)    

    B=np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
    N_mat=np.zeros((3,m_P),dtype=np.float64) # matrix  
    rhs=np.zeros(Nfem,dtype=np.float64)     # right hand side of Ax=b

    counter=0
    for iel in range(0,nel):

        f_el=np.zeros((ndof_V_el),dtype=np.float64)
        K_el=np.zeros((ndof_V_el,ndof_V_el),dtype=np.float64)
        G_el=np.zeros((ndof_V_el,m_P),dtype=np.float64)
        h_el=np.zeros((m_P),dtype=np.float64)

        for iq in range(0,nqel):

            for i in range(0,m_V):
                dNdx=dNdx_V[iq,i] 
                dNdy=dNdy_V[iq,i] 
                B[0,2*i  ]=dNdx
                B[1,2*i+1]=dNdy
                B[2,2*i  ]=dNdy
                B[2,2*i+1]=dNdx

            K_el+=B.T.dot(C.dot(B))*etaq[iel,iq]*JxWq[iq]

            for i in range(0,m_V):
                f_el[ndof_V*i+1]+=N_V[iq,i]*JxWq[iq]*rhoq[iel,iq]*gy

            N_mat[0,0:m_P]=N_P[iq,0:m_P]
            N_mat[1,0:m_P]=N_P[iq,0:m_P]

            G_el-=B.T.dot(N_mat)*JxWq[iq]

        # end for iq

        G_el*=eta_ref/L_ref

        # impose b.c. 
        for ikk in range(0,ndof_V_el):
            m1=local_to_globalV[ikk,iel]
            if bc_fix_V[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,ndof_V_el):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
               K_el[ikk,:]=0
               K_el[:,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val_V[m1]
               h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
               G_el[ikk,:]=0

        # assemble matrix and right hand side
        for ikk in range(ndof_V_el):
            m1=local_to_globalV[ikk,iel]
            for jkk in range(ndof_V_el):
                VV_V[counter]=K_el[ikk,jkk]
                counter+=1
            for jkk in range(0,m_P):
                VV_V[counter]=G_el[ikk,jkk]
                counter+=1
                VV_V[counter]=G_el[ikk,jkk]
                counter+=1
            rhs[m1]+=f_el[ikk]
        for k2 in range(0,m_P):
            m2=icon_P[k2,iel]
            rhs[Nfem_V+m2]+=h_el[k2]

    return VV_V,rhs

###############################################################################
