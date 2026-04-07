###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np
import numba

###################################################################################################
# local numbering of nodes
# 3--6--2
# 7  8  5
# 0--4--1
###################################################################################################

#@numba.njit
def build_matrix_stokes(bignb_Stokes,bignb_K,bignb_MP,bignb_G,nel,nq_per_element,m_V,m_P,ndof_V,\
                        Nfem_V,Nfem,ndof_V_el,icon_V,icon_P,\
                        rhoq,etaq,JxWq,local_to_globalV,gxq,gyq,N_V,N_P,dNdr_V,dNdt_V,dNdr_P,dNdt_P,\
                        jcbi00q,jcbi01q,jcbi10q,jcbi11q,eta_ref,L_ref,bc_fix_V,bc_val_V,\
                        bot_element,top_element,bot_free_slip,top_free_slip,geometry,theta_V,
                        axisymmetric,xq):
    """
    Args:
    Returns:
    """

    if axisymmetric:
       B=np.zeros((4,ndof_V*m_V),dtype=np.float64)
       N_mat=np.zeros((4,m_P),dtype=np.float64) 
       C=np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]],dtype=np.float64)
    else:
       B=np.zeros((3,ndof_V*m_V),dtype=np.float64)
       N_mat=np.zeros((3,m_P),dtype=np.float64) 
       C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
       #C=np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64) 

    VV_Stokes=np.zeros(bignb_Stokes,dtype=np.float64)    
    rhs_Stokes=np.zeros(Nfem,dtype=np.float64) 
    VV_MP=np.zeros(bignb_MP,dtype=np.float64)    
    VV_K=np.zeros(bignb_K,dtype=np.float64)    
    VV_G=np.zeros(bignb_G,dtype=np.float64)    

    counter=0
    counter_P=0
    counter_MP=0
    counter_K=0
    counter_G=0

    for iel in range(0,nel):

        f_el=np.zeros((ndof_V_el),dtype=np.float64)
        K_el=np.zeros((ndof_V_el,ndof_V_el),dtype=np.float64)
        G_el=np.zeros((ndof_V_el,m_P),dtype=np.float64)
        h_el=np.zeros((m_P),dtype=np.float64)
        MP_el=np.zeros((m_P,m_P),dtype=np.float64)
        H_el=np.zeros((m_P,ndof_V_el),dtype=np.float64)
        aa_mat=np.zeros((m_P,2),dtype=np.float64)
        bb_mat=np.zeros((2,ndof_V*m_V),dtype=np.float64)

        if axisymmetric:
           for iq in range(0,nq_per_element):
               dNdx_V=jcbi00q[iel,iq]*dNdr_V[iq,:]+jcbi01q[iel,iq]*dNdt_V[iq,:]
               dNdz_V=jcbi10q[iel,iq]*dNdr_V[iq,:]+jcbi11q[iel,iq]*dNdt_V[iq,:]
               coeffq=2*np.pi*xq[iel,iq]
               for i in range(0,m_V):
                   B[0,2*i  ]=dNdx_V[i]
                   B[1,2*i  ]=N_V[iq,i]/xq[iel,iq]
                   B[2,2*i+1]=dNdz_V[i]
                   B[3,2*i  ]=dNdz_V[i]
                   B[3,2*i+1]=dNdx_V[i]
               K_el+=B.T.dot(C.dot(B))*etaq[iel,iq]*JxWq[iel,iq]*coeffq
               for i in range(0,m_V):
                   f_el[ndof_V*i  ]+=N_V[iq,i]*JxWq[iel,iq]*coeffq*rhoq[iel,iq]*gxq[iel,iq]
                   f_el[ndof_V*i+1]+=N_V[iq,i]*JxWq[iel,iq]*coeffq*rhoq[iel,iq]*gyq[iel,iq]
               N_mat[0,0:m_P]=N_P[iq,0:m_P]
               N_mat[1,0:m_P]=N_P[iq,0:m_P]
               N_mat[2,0:m_P]=N_P[iq,0:m_P]
               G_el-=B.T.dot(N_mat)*JxWq[iel,iq]*coeffq
           # end for iq
        else:
           for iq in range(0,nq_per_element):
               dNdx_V=jcbi00q[iel,iq]*dNdr_V[iq,:]+jcbi01q[iel,iq]*dNdt_V[iq,:]
               dNdz_V=jcbi10q[iel,iq]*dNdr_V[iq,:]+jcbi11q[iel,iq]*dNdt_V[iq,:]
               dNdx_P=jcbi00q[iel,iq]*dNdr_P[iq,:]+jcbi01q[iel,iq]*dNdt_P[iq,:]
               dNdz_P=jcbi10q[iel,iq]*dNdr_P[iq,:]+jcbi11q[iel,iq]*dNdt_P[iq,:]
               for i in range(0,m_V):
                   B[0,2*i  ]=dNdx_V[i]
                   B[1,2*i+1]=dNdz_V[i]
                   B[2,2*i  ]=dNdz_V[i]
                   B[2,2*i+1]=dNdx_V[i]
               K_el+=B.T.dot(C.dot(B))*etaq[iel,iq]*JxWq[iel,iq]
               for i in range(0,m_V):
                   f_el[ndof_V*i  ]+=N_V[iq,i]*JxWq[iel,iq]*rhoq[iel,iq]*gxq[iel,iq]
                   f_el[ndof_V*i+1]+=N_V[iq,i]*JxWq[iel,iq]*rhoq[iel,iq]*gyq[iel,iq]
               N_mat[0,0:m_P]=N_P[iq,0:m_P]
               N_mat[1,0:m_P]=N_P[iq,0:m_P]
               G_el-=B.T.dot(N_mat)*JxWq[iel,iq]
               MP_el+=np.outer(N_P[iq,:],N_P[iq,:])*JxWq[iel,iq]

               aa_mat[:,0]=dNdx_P[:]
               aa_mat[:,1]=dNdz_P[:]
               for i in range(0,m_V):
                   bb_mat[0,2*i  ]=N_V[iq,i]
                   bb_mat[1,2*i+1]=N_V[iq,i]
               H_el+=aa_mat@bb_mat*JxWq[iel,iq]
           # end for iq
        # end if axisymmetric 

        G_el*=eta_ref/L_ref

        if geometry=='quarter' or geometry=='half' or geometry=='eighth':
           if top_element[iel] and top_free_slip: # free slip on top boundary
              for i in [2,3,6]:
                  inode=icon_V[i,iel] 
                  if (not bc_fix_V[2*inode]) and (not bc_fix_V[2*inode+1]): # no bc applied on node
                     RR=np.eye(ndof_V_el,dtype=np.float64)
                     idofn=2*i
                     idoft=2*i+1
                     RR[idofn,idofn]= np.cos(theta_V[inode]) ; RR[idofn,idoft]=np.sin(theta_V[inode])
                     RR[idoft,idofn]=-np.sin(theta_V[inode]) ; RR[idoft,idoft]=np.cos(theta_V[inode])
                     K_el=RR.dot(K_el.dot(RR.T))
                     G_el=RR.dot(G_el)
                     f_el=RR.dot(f_el)
                     K_ref=K_el[idofn,idofn]
                     K_el[idofn,:]=0
                     K_el[:,idofn]=0
                     K_el[idofn,idofn]=K_ref
                     G_el[idofn,:]=0
                     f_el[idofn]=0

           if bot_element[iel] and bot_free_slip: # free slip on bottom boundary
              for i in [0,1,4]:
                  inode=icon_V[i,iel] 
                  if (not bc_fix_V[2*inode]) and (not bc_fix_V[2*inode+1]): # no bc applied on node
                     RR=np.eye(ndof_V_el,dtype=np.float64)
                     idofn=2*i
                     idoft=2*i+1
                     RR[idofn,idofn]= np.cos(theta_V[inode]) ; RR[idofn,idoft]=np.sin(theta_V[inode])
                     RR[idoft,idofn]=-np.sin(theta_V[inode]) ; RR[idoft,idoft]=np.cos(theta_V[inode])
                     K_el=RR.dot(K_el.dot(RR.T))
                     G_el=RR.dot(G_el)
                     f_el=RR.dot(f_el)
                     K_ref=K_el[idofn,idofn]
                     K_el[idofn,:]=0
                     K_el[:,idofn]=0
                     K_el[idofn,idofn]=K_ref
                     G_el[idofn,:]=0
                     f_el[idofn]=0

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

        # assemble Stokes matrix and right hand side
        for ikk in range(ndof_V_el):
            for jkk in range(ndof_V_el):
                VV_Stokes[counter]=K_el[ikk,jkk]
                counter+=1
            for jkk in range(0,m_P):
                VV_Stokes[counter]=G_el[ikk,jkk]
                counter+=1
                VV_Stokes[counter]=G_el[ikk,jkk]
                counter+=1
            m1=local_to_globalV[ikk,iel]
            rhs_Stokes[m1]+=f_el[ikk]
        for k2 in range(0,m_P):
            m2=icon_P[k2,iel]
            rhs_Stokes[Nfem_V+m2]+=h_el[k2]

        # assemble K block 
        for ikk in range(ndof_V_el):
            for jkk in range(ndof_V_el):
                VV_K[counter_K]=K_el[ikk,jkk]
                counter_K+=1

        # assemble G block 
        for ikk in range(ndof_V_el):
            for jkk in range(0,m_P):
                VV_G[counter_G]=G_el[ikk,jkk]
                counter_G+=1

        # assemble pressure mass matrix
        for k1 in range(0,m_P):
            for k2 in range(0,m_P):
                VV_MP[counter_MP]=MP_el[k1,k2]
                counter_MP+=1

    return VV_Stokes,rhs_Stokes,VV_K,VV_G,VV_MP

###################################################################################################
