###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################


def assess_steady_state(
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
    tol_ss: float,
):

    if solve_Stokes:
        steady_state_u = np.linalg.norm(u_mem - u, 2) / np.linalg.norm(u, 2) < tol_ss
        steady_state_w = np.linalg.norm(w_mem - w, 2) / np.linalg.norm(w, 2) < tol_ss
        steady_state_p = np.linalg.norm(p_mem - p, 2) / np.linalg.norm(p, 2) < tol_ss
    else:
        steady_state_u = False
        steady_state_w = False
        steady_state_p = False

    if solve_T:
        steady_state_T = np.linalg.norm(T_mem - T, 2) / np.linalg.norm(T, 2) < tol_ss
        print(
            "     -> u,w,p,T",
            steady_state_u,
            steady_state_w,
            steady_state_p,
            steady_state_T,
        )
    else:
        steady_state_T = True
        print("     -> u,w,p", steady_state_u, steady_state_w, steady_state_p)

    steady_state = steady_state_u and steady_state_w and steady_state_p and steady_state_T

    u_mem = u.copy()
    w_mem = w.copy()
    p_mem = p.copy()

    if solve_T:
        T_mem = T.copy()

    return steady_state, u_mem, w_mem, p_mem, T_mem


###################################################################################################
