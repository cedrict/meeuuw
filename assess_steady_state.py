###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################


def assess_nlconvergence(
    istep: int,
    iter_nl: int,
    solve_Stokes: bool,
    solve_T: bool,
    u: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    T: np.ndarray,
    u_mem: np.ndarray,
    w_mem: np.ndarray,
    p_mem: np.ndarray,
    T_mem: np.ndarray,
    tol_nl: float,
    inside_nonlinear_iterations,
    conv_file,
):

    if solve_Stokes:
        xi_u=np.linalg.norm(u_mem - u, 2) / np.linalg.norm(u, 2)
        xi_w=np.linalg.norm(w_mem - w, 2) / np.linalg.norm(w, 2)
        xi_p=np.linalg.norm(p_mem - p, 2) / np.linalg.norm(p, 2)
        nlconvergence_u = xi_u < tol_nl
        nlconvergence_w = xi_w < tol_nl
        nlconvergence_p = xi_p < tol_nl
    else:
        xi_u=0
        xi_w=0
        xi_p=0
        nlconvergence_u = False
        nlconvergence_w = False
        nlconvergence_p = False

    if solve_T:
        xi_T=np.linalg.norm(T_mem - T, 2) / np.linalg.norm(T, 2)
        nlconvergence_T = xi_T < tol_nl
    else:
        xi_T=0
        nlconvergence_T = True

    print("     -> NL: istep %d iter_nl %d |xi_u %.3e |xi_w %.3e |xi_p %.3e |xi_T %.3e |tol %.3e " %\
         (istep,iter_nl,xi_u,xi_w,xi_p,xi_T,tol_nl))

    conv_file.write("%.3e %.3e %.3e %.3e %.3e %.3e \n" % (istep+iter_nl/200,xi_u,xi_w,xi_p,xi_T,tol_nl))
    conv_file.flush()

    print(
        "     -> NL: u,w,p,T",
        nlconvergence_u,
        nlconvergence_w,
        nlconvergence_p,
        nlconvergence_T,
    )


    nlconvergence = nlconvergence_u and nlconvergence_w and nlconvergence_p and nlconvergence_T

    if nlconvergence:
       print("     -> NL CONVERGED!")
       inside_nonlinear_iterations=False
    else:
       print("     -> NL inside_nonlinear_iterations=",inside_nonlinear_iterations)

    u_mem = u.copy()
    w_mem = w.copy()
    p_mem = p.copy()

    if solve_T and not inside_nonlinear_iterations:
        T_mem = T.copy()

    return u_mem, w_mem, p_mem, T_mem, inside_nonlinear_iterations


###################################################################################################
