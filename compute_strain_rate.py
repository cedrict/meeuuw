###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np
import numba
import scipy.sparse as sps
from basis_functions import *
from scipy import sparse

###################################################################################################
# this function computes the strain rate in the middle of each element
###################################################################################################

@numba.njit
def compute_elemental_strain_rate(icon_V,u,v,nn_V,nel,x_V,y_V):

    exx=np.zeros(nel,dtype=np.float64)  
    eyy=np.zeros(nel,dtype=np.float64)  
    exy=np.zeros(nel,dtype=np.float64)  
    jcb=np.zeros((2,2),dtype=np.float64)
    jcbi=np.zeros((2,2),dtype=np.float64)

    rq=sq=0.
    dNdr=basis_functions_V_dr(rq,sq)
    dNdt=basis_functions_V_dt(rq,sq)

    for iel in range(0,nel):
        jcb[0,0]=np.dot(dNdr,x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr,y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNdt,x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNdt,y_V[icon_V[:,iel]])
        jcbi=np.linalg.inv(jcb)
        dNdx=jcbi[0,0]*dNdr+jcbi[0,1]*dNdt
        dNdy=jcbi[1,0]*dNdr+jcbi[1,1]*dNdt
        exx[iel]=np.dot(dNdx,u[icon_V[:,iel]])
        eyy[iel]=np.dot(dNdy,v[icon_V[:,iel]])
        exy[iel]=np.dot(dNdx,v[icon_V[:,iel]])*0.5+\
                 np.dot(dNdy,u[icon_V[:,iel]])*0.5

    return exx,eyy,exy

###################################################################################################
# this function computes the strain rate on the V nodes, using corners -> node averaging approach
###################################################################################################

@numba.njit
def compute_nodal_strain_rate(icon_V,u,v,nn_V,m_V,nel,dNdr_V_n,dNds_V_n,jcbi00n,jcbi01n,jcbi10n,jcbi11n):

    count=np.zeros(nn_V,dtype=np.float64)  
    exx_n=np.zeros(nn_V,dtype=np.float64)  
    eyy_n=np.zeros(nn_V,dtype=np.float64)  
    exy_n=np.zeros(nn_V,dtype=np.float64)  

    for iel in range(0,nel):
        for i in range(0,m_V):
            inode=icon_V[i,iel]
            dNdx=jcbi00n[iel,i]*dNdr_V_n[i,:]+jcbi01n[iel,i]*dNds_V_n[i,:]
            dNdy=jcbi10n[iel,i]*dNdr_V_n[i,:]+jcbi11n[iel,i]*dNds_V_n[i,:]
            exx_n[inode]+=np.dot(dNdx,u[icon_V[:,iel]])
            eyy_n[inode]+=np.dot(dNdy,v[icon_V[:,iel]])
            exy_n[inode]+=0.5*np.dot(dNdx,v[icon_V[:,iel]])+\
                          0.5*np.dot(dNdy,u[icon_V[:,iel]])
            count[inode]+=1
        #end for
    #end for
    exx_n/=count
    eyy_n/=count
    exy_n/=count

    return exx_n,eyy_n,exy_n

###################################################################################################
# this function computes the strain rate on the V nodes, using a more FEM consistent way 
# this cannot be numba'ed, bc sparse stuff will not allow it.
# if needed we should take the sparse+solve outside
###################################################################################################

##@numba.njit
def compute_nodal_strain_rate2(bignb,II,JJ,m_T,nq_per_element,icon_V,u,v,nn_V,nel,JxWq,N_V,dNdr_V,dNds_V,\
                               jcbi00q,jcbi01q,jcbi10q,jcbi11q):

    VV_T=np.zeros(bignb,dtype=np.float64)    
    rhs_xx=np.zeros(nn_V,dtype=np.float64)
    rhs_yy=np.zeros(nn_V,dtype=np.float64)
    rhs_xy=np.zeros(nn_V,dtype=np.float64)

    counter=0
    for iel in range(0,nel):
        A_el=np.zeros((m_T,m_T),dtype=np.float64)
        bxx_el=np.zeros(m_T,dtype=np.float64)
        byy_el=np.zeros(m_T,dtype=np.float64)
        bxy_el=np.zeros(m_T,dtype=np.float64)

        for iq in range(0,nq_per_element):
            dNdx=jcbi00q[iel,iq]*dNdr_V[iq,:]+jcbi01q[iel,iq]*dNds_V[iq,:]
            dNdy=jcbi10q[iel,iq]*dNdr_V[iq,:]+jcbi11q[iel,iq]*dNds_V[iq,:]
            N=N_V[iq,:]
            A_el+=np.outer(N,N)*JxWq[iel,iq] 
            exxq=np.dot(dNdx,u[icon_V[:,iel]])
            eyyq=np.dot(dNdy,v[icon_V[:,iel]])
            exyq=0.5*np.dot(dNdx,v[icon_V[:,iel]])\
                +0.5*np.dot(dNdy,u[icon_V[:,iel]])
            bxx_el+=N*exxq*JxWq[iel,iq]
            byy_el+=N*eyyq*JxWq[iel,iq]
            bxy_el+=N*exyq*JxWq[iel,iq]
        #end for

        # assemble matrix K_mat and right hand side rhs
        for ikk in range(m_T):
            m1=icon_V[ikk,iel]
            for jkk in range(m_T):
                VV_T[counter]=A_el[ikk,jkk]
                counter+=1
            rhs_xx[m1]+=bxx_el[ikk]
            rhs_yy[m1]+=byy_el[ikk]
            rhs_xy[m1]+=bxy_el[ikk]
        #end for

    #end iel

    sparse_matrix=sparse.coo_matrix((VV_T,(II,JJ)),shape=(nn_V,nn_V)).tocsr()

    exx_n=sps.linalg.spsolve(sparse_matrix,rhs_xx)
    eyy_n=sps.linalg.spsolve(sparse_matrix,rhs_yy)
    exy_n=sps.linalg.spsolve(sparse_matrix,rhs_xy)

    return exx_n,eyy_n,exy_n,e_n

###################################################################################################
