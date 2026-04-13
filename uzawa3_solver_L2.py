###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as sla

###################################################################################################
# the function implicitely assumes matrices K,Mp in csc format (necessary for splu)
# the numpy.linalg.norm uses the 2-norm by default. 
# the scipy.sparse.linalg.splu computes the LU decomposition of a sparse, square matrix. 
# splu stands for SuperLU. https://portal.nersc.gov/project/sparse/superlu/
# scipy.sparse.linalg.cg: Solve Ax = b with the Conjugate Gradient method, for a SPD matrix A.
# no preconditioner yet
# TODO: use lith pressure as P guess?
# I have tried CG as innersolve -> too slow, even for constant viscosity ms
# I have tried cg on Mp -> fail?!
###################################################################################################

def uzawa3_solver_L2(K_mat,G_mat,GT_mat,MP_mat,H_mat,f_rhs,h_rhs,Nfem_P):

   innerMp='splu'
   innerK='splu'
   tol=1e-6
   niter=100
   array_xiV=np.zeros(niter,dtype=np.float64) 
   array_xiP=np.zeros(niter,dtype=np.float64) 
   array_alpha=np.zeros(niter,dtype=np.float64) 

   solP=np.zeros(Nfem_P,dtype=np.float64) #  guess pressure is zero.

   # compute V_0
   if innerK=='direct':
      solV=sps.linalg.spsolve(K_mat,f_rhs)  
   elif innerK=='splu':
      LU=sla.splu(K_mat)
      solV=LU.solve(f_rhs)
   elif innerK=='cg':
      solV=sps.linalg.cg(K_mat,f_rhs)[0]

   # compute r_0
   if innerMp=='direct':
      rvect_k=sps.linalg.spsolve(MP_mat,GT_mat.dot(solV)-h_rhs)    
   elif innerMp=='splu':
      LU_Mp=sla.splu(MP_mat)
      rvect_k=LU_Mp.solve(GT_mat.dot(solV)-h_rhs)            

   pvect_k=np.copy(rvect_k)                                         # compute p_0

   for k in range (0,niter): #--------------------------------------#
                                                                    # 
       #AAA                                                         #
       if innerK=='direct':                                         #
          dvect_k=sps.linalg.spsolve(K_mat,G_mat.dot(pvect_k))      #
       elif innerK=='splu':                                         #
          dvect_k=LU.solve(G_mat.dot(pvect_k))                      #
       elif innerK=='cg':                                           #
          ptildevect_k=G_mat.dot(pvect_k)                           #
          rhsmax=np.max(ptildevect_k)                               #
          dvect_k=sps.linalg.cg(K_mat,ptildevect_k/rhsmax)[0]       #
          dvect_k*=rhsmax                                           #
                                                                    #
       #BBB                                                         #
       numerator=rvect_k.dot(MP_mat.dot(rvect_k))                   #
       denominator=pvect_k.dot(H_mat.dot(dvect_k))                  #
       alpha=numerator/denominator                                  #
       array_alpha[k]=alpha                                         #
       #CCC                                                         #
       solP+=alpha*pvect_k                                          #
       #DDD                                                         #
       solV-=alpha*dvect_k                                          #
       #EEE                                                         #
                                                                    #
       if innerMp=='direct':                                        #
          dr=sps.linalg.spsolve(MP_mat,-alpha*GT_mat.dot(dvect_k))  #
       elif innerMp=='splu':                                        #
          dr=LU_Mp.solve(-alpha*GT_mat.dot(dvect_k))                #

       rvect_kp1=rvect_k+dr                                         #
       #FFF                                                         #
       numerator=rvect_kp1.dot(MP_mat.dot(rvect_kp1))               #
       denominator=rvect_k.dot(MP_mat.dot(rvect_k))                 #
       beta=numerator/denominator                                   #
       #GGG                                                         #
       pvect_kp1=rvect_kp1+beta*pvect_k                             #
                                                                    #
       xiP=np.linalg.norm(alpha*pvect_k)  # i.e. norm of (Pk+1-Pk)  #
       xiV=np.linalg.norm(alpha*dvect_k)  # i.e. norm of (Vk+1-Vk)  #
       array_xiV[k]=xiV                                             #
       array_xiP[k]=xiP                                             #
       if xiP<tol and xiV<tol: break                                #
                                                                    #
       rvect_k=np.copy(rvect_kp1)                                   #
       pvect_k=np.copy(pvect_kp1)                                   #
                                                                    #
   #end for k #-----------------------------------------------------#
    
   return solV,solP,k+1,array_xiV,array_xiP,array_alpha

############################################################################### 
