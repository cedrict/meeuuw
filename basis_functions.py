###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numba
import numpy as np

m_V = 9  # number of velocity nodes per element
m_P = 4  # number of pressure nodes per element
m_T = 9  # number of temperature nodes per element

###################################################################################################
# Q2 velocity basis functions
###################################################################################################

r_V = [-1, 1, 1, -1, 0, 1, 0, -1, 0]
t_V = [-1, -1, 1, 1, -1, 0, 1, 0, 0]

@numba.njit
def basis_functions_V(r: float, s: float):
    N0 = 0.5 * r * (r - 1.0) * 0.5 * s * (s - 1.0)
    N1 = 0.5 * r * (r + 1.0) * 0.5 * s * (s - 1.0)
    N2 = 0.5 * r * (r + 1.0) * 0.5 * s * (s + 1.0)
    N3 = 0.5 * r * (r - 1.0) * 0.5 * s * (s + 1.0)
    N4 = (1.0 - r**2) * 0.5 * s * (s - 1.0)
    N5 = 0.5 * r * (r + 1.0) * (1.0 - s**2)
    N6 = (1.0 - r**2) * 0.5 * s * (s + 1.0)
    N7 = 0.5 * r * (r - 1.0) * (1.0 - s**2)
    N8 = (1.0 - r**2) * (1.0 - s**2)
    return np.array([N0, N1, N2, N3, N4, N5, N6, N7, N8], dtype=np.float64)


@numba.njit
def basis_functions_V_dr(r: float, s: float):
    dNdr0 = 0.5 * (2.0 * r - 1.0) * 0.5 * s * (s - 1)
    dNdr1 = 0.5 * (2.0 * r + 1.0) * 0.5 * s * (s - 1)
    dNdr2 = 0.5 * (2.0 * r + 1.0) * 0.5 * s * (s + 1)
    dNdr3 = 0.5 * (2.0 * r - 1.0) * 0.5 * s * (s + 1)
    dNdr4 = (-2.0 * r) * 0.5 * s * (s - 1)
    dNdr5 = 0.5 * (2.0 * r + 1.0) * (1.0 - s**2)
    dNdr6 = (-2.0 * r) * 0.5 * s * (s + 1)
    dNdr7 = 0.5 * (2.0 * r - 1.0) * (1.0 - s**2)
    dNdr8 = (-2.0 * r) * (1.0 - s**2)
    return np.array([dNdr0, dNdr1, dNdr2, dNdr3, dNdr4, dNdr5, dNdr6, dNdr7, dNdr8], dtype=np.float64)


@numba.njit
def basis_functions_V_dt(r: float, s: float):
    dNdt0 = 0.5 * r * (r - 1.0) * 0.5 * (2.0 * s - 1.0)
    dNdt1 = 0.5 * r * (r + 1.0) * 0.5 * (2.0 * s - 1.0)
    dNdt2 = 0.5 * r * (r + 1.0) * 0.5 * (2.0 * s + 1.0)
    dNdt3 = 0.5 * r * (r - 1.0) * 0.5 * (2.0 * s + 1.0)
    dNdt4 = (1.0 - r**2) * 0.5 * (2.0 * s - 1.0)
    dNdt5 = 0.5 * r * (r + 1.0) * (-2.0 * s)
    dNdt6 = (1.0 - r**2) * 0.5 * (2.0 * s + 1.0)
    dNdt7 = 0.5 * r * (r - 1.0) * (-2.0 * s)
    dNdt8 = (1.0 - r**2) * (-2.0 * s)
    return np.array([dNdt0, dNdt1, dNdt2, dNdt3, dNdt4, dNdt5, dNdt6, dNdt7, dNdt8], dtype=np.float64)

###################################################################################################
# Q2 temperature basis functions
###################################################################################################

r_T = [-1, 1, 1, -1, 0, 1, 0, -1, 0]
t_T = [-1, -1, 1, 1, -1, 0, 1, 0, 0]

@numba.njit
def basis_functions_T(r: float, s: float):
    N0 = 0.5 * r * (r - 1.0) * 0.5 * s * (s - 1.0)
    N1 = 0.5 * r * (r + 1.0) * 0.5 * s * (s - 1.0)
    N2 = 0.5 * r * (r + 1.0) * 0.5 * s * (s + 1.0)
    N3 = 0.5 * r * (r - 1.0) * 0.5 * s * (s + 1.0)
    N4 = (1.0 - r**2) * 0.5 * s * (s - 1.0)
    N5 = 0.5 * r * (r + 1.0) * (1.0 - s**2)
    N6 = (1.0 - r**2) * 0.5 * s * (s + 1.0)
    N7 = 0.5 * r * (r - 1.0) * (1.0 - s**2)
    N8 = (1.0 - r**2) * (1.0 - s**2)
    return np.array([N0, N1, N2, N3, N4, N5, N6, N7, N8], dtype=np.float64)


@numba.njit
def basis_functions_T_dr(r: float, s: float):
    dNdr0 = 0.5 * (2.0 * r - 1.0) * 0.5 * s * (s - 1)
    dNdr1 = 0.5 * (2.0 * r + 1.0) * 0.5 * s * (s - 1)
    dNdr2 = 0.5 * (2.0 * r + 1.0) * 0.5 * s * (s + 1)
    dNdr3 = 0.5 * (2.0 * r - 1.0) * 0.5 * s * (s + 1)
    dNdr4 = (-2.0 * r) * 0.5 * s * (s - 1)
    dNdr5 = 0.5 * (2.0 * r + 1.0) * (1.0 - s**2)
    dNdr6 = (-2.0 * r) * 0.5 * s * (s + 1)
    dNdr7 = 0.5 * (2.0 * r - 1.0) * (1.0 - s**2)
    dNdr8 = (-2.0 * r) * (1.0 - s**2)
    return np.array([dNdr0, dNdr1, dNdr2, dNdr3, dNdr4, dNdr5, dNdr6, dNdr7, dNdr8], dtype=np.float64)


@numba.njit
def basis_functions_T_dt(r: float, s: float):
    dNdt0 = 0.5 * r * (r - 1.0) * 0.5 * (2.0 * s - 1.0)
    dNdt1 = 0.5 * r * (r + 1.0) * 0.5 * (2.0 * s - 1.0)
    dNdt2 = 0.5 * r * (r + 1.0) * 0.5 * (2.0 * s + 1.0)
    dNdt3 = 0.5 * r * (r - 1.0) * 0.5 * (2.0 * s + 1.0)
    dNdt4 = (1.0 - r**2) * 0.5 * (2.0 * s - 1.0)
    dNdt5 = 0.5 * r * (r + 1.0) * (-2.0 * s)
    dNdt6 = (1.0 - r**2) * 0.5 * (2.0 * s + 1.0)
    dNdt7 = 0.5 * r * (r - 1.0) * (-2.0 * s)
    dNdt8 = (1.0 - r**2) * (-2.0 * s)
    return np.array([dNdt0, dNdt1, dNdt2, dNdt3, dNdt4, dNdt5, dNdt6, dNdt7, dNdt8], dtype=np.float64)


###################################################################################################
# Q1 pressure basis functions
###################################################################################################


@numba.njit
def basis_functions_P(r: float, s: float):
    N0 = 0.25 * (1 - r) * (1 - s)
    N1 = 0.25 * (1 + r) * (1 - s)
    N2 = 0.25 * (1 + r) * (1 + s)
    N3 = 0.25 * (1 - r) * (1 + s)
    return np.array([N0, N1, N2, N3], dtype=np.float64)


@numba.njit
def basis_functions_P_dr(r:float, s:float):
    dNdr0 = -0.25 * (1 - s)
    dNdr1 = 0.25 * (1 - s)
    dNdr2 = 0.25 * (1 + s)
    dNdr3 = -0.25 * (1 + s)
    return np.array([dNdr0, dNdr1, dNdr2, dNdr3], dtype=np.float64)


@numba.njit
def basis_functions_P_dt(r:float, s: float):
    dNdt0 = -0.25 * (1 - r)
    dNdt1 = -0.25 * (1 + r)
    dNdt2 = 0.25 * (1 + r)
    dNdt3 = 0.25 * (1 - r)
    return np.array([dNdt0, dNdt1, dNdt2, dNdt3], dtype=np.float64)


###################################################################################################
