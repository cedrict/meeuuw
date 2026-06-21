###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numba
import numpy as np

m_V = 5  # number of velocity nodes per element
m_P = 4  # number of pressure nodes per element
m_T = 4  # number of temperature nodes per element

###################################################################################################
# Q1+ velocity basis functions
###################################################################################################

r_V = [-1, 1, 1, -1, 0]
t_V = [-1, -1, 1, 1, 0]

beta = 0.25


def B(r, s):
    return (1 - r**2) * (1 - s**2) * (1 + beta * (r + s))


def dBdr(r, s):
    return (s**2 - 1) * (-beta + 3 * beta * r**2 + 2 * r * (beta * s + 1))


def dBds(r, s):
    return (r**2 - 1) * (-beta + 2 * s * (beta * r + 1) + 3 * beta * s**2)


@numba.njit
def basis_functions_V(r: float, s: float):
    N0 = 0.25 * (1 - r) * (1 - s) - 0.25 * B(r, s)
    N1 = 0.25 * (1 + r) * (1 - s) - 0.25 * B(r, s)
    N2 = 0.25 * (1 + r) * (1 + s) - 0.25 * B(r, s)
    N3 = 0.25 * (1 - r) * (1 + s) - 0.25 * B(r, s)
    N4 = B(r, s)
    return np.array([N0, N1, N2, N3, N4], dtype=np.float64)


@numba.njit
def basis_functions_V_dr(r: float, s: float):
    dNdr0 = -0.25 * (1.0 - s) - 0.25 * dBdr(r, s)
    dNdr1 = +0.25 * (1.0 - s) - 0.25 * dBdr(r, s)
    dNdr2 = +0.25 * (1.0 + s) - 0.25 * dBdr(r, s)
    dNdr3 = -0.25 * (1.0 + s) - 0.25 * dBdr(r, s)
    dNdr4 = dBdr(r, s)
    return np.array([dNdr0, dNdr1, dNdr2, dNdr3, dNdr4], dtype=np.float64)


@numba.njit
def basis_functions_V_dt(r: float, s: float):
    dNdt0 = -0.25 * (1.0 - r) - 0.25 * dBds(r, s)
    dNdt1 = -0.25 * (1.0 + r) - 0.25 * dBds(r, s)
    dNdt2 = +0.25 * (1.0 + r) - 0.25 * dBds(r, s)
    dNdt3 = +0.25 * (1.0 - r) - 0.25 * dBds(r, s)
    dNdt4 = dBds(r, s)
    return np.array([dNdt0, dNdt1, dNdt2, dNdt3, dNdt4], dtype=np.float64)


###################################################################################################
# Q1 temperature basis functions
###################################################################################################

r_T = [-1, 1, 1, -1]
t_T = [-1, -1, 1, 1]


@numba.njit
def basis_functions_T(r: float, s: float):
    N0 = 0.25 * (1 - r) * (1 - s)
    N1 = 0.25 * (1 + r) * (1 - s)
    N2 = 0.25 * (1 + r) * (1 + s)
    N3 = 0.25 * (1 - r) * (1 + s)
    return np.array([N0, N1, N2, N3], dtype=np.float64)


@numba.njit
def basis_functions_T_dr(r: float, s: float):
    dNdr0 = -0.25 * (1 - s)
    dNdr1 = 0.25 * (1 - s)
    dNdr2 = 0.25 * (1 + s)
    dNdr3 = -0.25 * (1 + s)
    return np.array([dNdr0, dNdr1, dNdr2, dNdr3], dtype=np.float64)


@numba.njit
def basis_functions_T_dt(r: float, s: float):
    dNdt0 = -0.25 * (1 - r)
    dNdt1 = -0.25 * (1 + r)
    dNdt2 = 0.25 * (1 + r)
    dNdt3 = 0.25 * (1 - r)
    return np.array([dNdt0, dNdt1, dNdt2, dNdt3], dtype=np.float64)


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
def basis_functions_P_dr(r: float, s: float):
    dNdr0 = -0.25 * (1 - s)
    dNdr1 = 0.25 * (1 - s)
    dNdr2 = 0.25 * (1 + s)
    dNdr3 = -0.25 * (1 + s)
    return np.array([dNdr0, dNdr1, dNdr2, dNdr3], dtype=np.float64)


@numba.njit
def basis_functions_P_dt(r: float, s: float):
    dNdt0 = -0.25 * (1 - r)
    dNdt1 = -0.25 * (1 + r)
    dNdt2 = 0.25 * (1 + r)
    dNdt3 = 0.25 * (1 - r)
    return np.array([dNdt0, dNdt1, dNdt2, dNdt3], dtype=np.float64)


###################################################################################################
