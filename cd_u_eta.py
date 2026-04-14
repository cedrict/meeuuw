###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as sla

###################################################################################################
# never tested with h not zero
###################################################################################################

def cd_u_eta(K,G,GT,M,M_eta,H,f,h,Nfem_P):

   innerM='splu'
   innerK='splu'
   tol=1e-6
   niter=100
   array_xiV=np.zeros(niter,dtype=np.float64) 
   array_xiP=np.zeros(niter,dtype=np.float64) 
   array_alpha=np.zeros(niter,dtype=np.float64) 

   #(A) Initialization
   solP=np.zeros(Nfem_P,dtype=np.float64)

   #(B) Initial solve
   if innerK=='direct':
      solV=sps.linalg.spsolve(K,f)  
   elif innerK=='splu':
      LU_K=sla.splu(K)
      solV=LU_K.solve(f)

   #(C) Initial residual
   if innerM=='direct': 
      q_k=sps.linalg.spsolve(M,-GT@solV+M@h)
   elif innerM=='splu':
      LU_M=sla.splu(M)
      LU_M_eta=sla.splu(M_eta)
      q_k=LU_M.solve(-GT@solV+M@h)

   #(D) Preconditioning
   w_k=sps.linalg.spsolve(M_eta,-GT@solV+M@h)    
   d_k=-np.copy(w_k)

   for k in range (0,niter): #--------------------------------------#
                                                                    #
       #(E) Direction solve                                         #
       if innerK=='direct':                                         #
          z_k=sps.linalg.spsolve(K,G@d_k)                           #
       elif innerK=='splu':                                         #
          z_k=LU_K.solve(G@d_k)                                     #
                                                                    #
       #(F) Step-size selection                                     # 
       numerator=q_k.dot(M.dot(w_k))                                #
       denominator=d_k.dot(H.dot(z_k))                              #
       alpha=numerator/denominator           ; array_alpha[k]=alpha #
                                                                    #
       #(G) Pressure update                                         #
       solP+=alpha*d_k                                              #
                                                                    #
       #(H) Velocity update                                         #
       solV-=alpha*z_k                                              #
                                                                    #
       #(I) Residual update                                         #
       if innerM=='direct':                                         # 
          q_kp1=sps.linalg.spsolve(M,-GT@solV+M@h)                  #
       elif innerM=='splu':                                         #
          q_kp1=LU_M.solve(-GT@solV+M@h)                            #
                                                                    #
       #(J) Preconditioning                                         #
       if innerM=='direct':                                         #
          w_kp1=sps.linalg.spsolve(M_eta,M@q_kp1)                   #
       elif innerM=='splu':                                         #
          w_kp1=LU_M_eta.solve(M@q_kp1)                             #
                                                                    #
       #(K) Conjugate direction                                     #
       numerator=q_kp1.dot(M.dot(w_kp1))                            #
       denominator=q_k.dot(M.dot(w_k))                              #
       beta=numerator/denominator                                   #
       d_kp1=-w_kp1+beta*d_k                                        #
                                                                    #
       xiP=np.linalg.norm(alpha*d_k)             ; array_xiP[k]=xiP #
       xiV=np.linalg.norm(alpha*z_k)             ; array_xiV[k]=xiV #
                                                                    #
       if xiP<tol and xiV<tol: break                                #
                                                                    #
       w_k=np.copy(w_kp1)                                           #
       d_k=np.copy(d_kp1)                                           #
       q_k=np.copy(q_kp1)                                           #
                                                                    #
   #end for k #-----------------------------------------------------#
    
   return solV,solP,k+1,array_xiV,array_xiP,array_alpha

############################################################################### 
