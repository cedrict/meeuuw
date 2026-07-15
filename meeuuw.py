###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np
import sys as sys
import numba
import time as clock
import scipy.sparse as sps
import argparse
import os

import global_stuff
from apply_mesh_stretching_x import *
from apply_mesh_stretching_z import *
from assess_steady_state import *
from basis_functions_setup_q import *
from basis_functions_setup_Vnodes import *
from basis_functions_setup_Tnodes import *
from basis_functions import *
from build_matrix_plith import *
from build_matrix_stokes import *
from build_matrix_energy import *
from build_velocity_mesh import *
from build_velocity_connectivity import *
from build_pressure_mesh import *
from build_pressure_connectivity import *
from define_mapping import *
from evolve_mesh import *
from compute_domain_volume import *
from compute_normals import *
from compute_strain_rate import *
from compute_avrg_profiles import *
from compute_nodal_heat_flux import *
from compute_nodal_pressure import *
from compute_domain_volume import *
from compute_center_coordinates import *
from compute_gravity_at_point import *
from compute_gravity_fromDT_at_point import *
from compute_pressure_offset import *
from compute_nodal_pressure_gradient import *
from compute_full_stress_tensor import *
from compute_deviatoric_stress_tensor import *
from constants import *
from cd_u_eta import *
from locate_particles import *
from open_files import *
from output_final_profiles import *
from output_swarm_to_vtu import *
from output_swarm_to_png import *
from output_swarm_to_ascii import *
from output_solution_to_vtu import *
from output_solution_to_png import *
from output_quadpoints_to_vtu import *
from output_test_results import *
from output_mesh_V import *
from pic_functions import *
from prescribe_initial_topography import *
from print_timings import *
from process_velocity_solution_vectors import *
from project_nodal_field_onto_qpoints import *
from postprocessors import *
from PoissonDisc import *
from quadrature_setup import *
from remove_net_rotation import *
from scipy import sparse
from swarm_coordinates_setup import *
from straighten_edges_axisymmetric import *
from toolbox import *
from update_F import *
from uzawa3_solver_L2 import *
from write_in_pvd_files import *

print("======================================================")
print("----------------------- MEEUUW -----------------------")
print("======================================================")

###############################################################################
# set lots of generic parameters to default value
###############################################################################

from set_default_parameters import *

###############################################################################
# experiment  0: Blankenbach et al, 1993 - isoviscous convection
# experiment  1: van Keken et al, JGR, 1997 - Rayleigh-Taylor experiment
# experiment  2: Schmeling et al, PEPI 2008 - Newtonian subduction
# experiment  3: Tosi et al, 2015 - visco-plastic convection
# experiment  4: Lindi MSc
# experiment  5: Trompert & Hansen, Nature 1998 - convection w/ plate-like
# experiment  6: Crameri et al, GJI 2012 (cosine perturbation & plume)
# experiment  7: ESA workshop
# experiment  8: quarter - sinker
# experiment  9: axisymmetric Mars setup
# experiment 10: axisymmetric 4D dyn Earth benchmark of Stokes sphere
# experiment 11: rising plume
# experiment 12: hollow earth gravity benchmark
# experiment 13: sinking block benchmark
# experiment 14: slab detachment (Schmalholz 2011)
# experiment 15: stokes sphere axisymmetric
# experiment 16: subduction initiation from Matsumoto and Tomoda (1983)
# experiment 17: sinking sphere 512km with sticky air
# experiment 18: sinking sphere unit square with sticky air
# experiment 19: Donea & Huerta manufactured solution
# experiment 20: sand box for paper review (SHOULD BE ...?)
# experiment 21: SolCx
# experiment 22: SolKz
# experiment 23: cohf19 experiment / Sylas
# experiment 24: murphy & king bsc thesis
# experiment 25: BA vs EBA box
# experiment 26: plume-lithosphere wali21
# experiment 27: folding 
# experiment 28: Lithospheric Drip based on bagu25
###############################################################################

experiment = 19

parser = argparse.ArgumentParser()
parser.add_argument("--nelx", type=int, default=0)
parser.add_argument("--nelz", type=int, default=0)
parser.add_argument("--nstep", type=int, default=0)
parser.add_argument("--nq_per_dim", type=int, default=0)
parser.add_argument("--e", type=int, default=-1)
parser.add_argument("--axisymmetric", type=int, default=0)
parser.add_argument("--straighten_edges", type=int, default=0)
parser.add_argument("--remove_rho_profile", type=int, default=0)
parser.add_argument("--particle_distribution", type=int, default=-1, choices=[0, 1, 2, 3])
parser.add_argument("--nparticle_per_dim", type=int, default=0)
parser.add_argument("--averaging", type=int, default=0, choices=[1, 2, 3])
parser.add_argument("--nodal_projection_type", type=int, default=0, choices=[1, 2, 3, 4])
parser.add_argument("--particle_rho_projection", type=int, default=-1)
parser.add_argument("--particle_eta_projection", type=int, default=-1)
parser.add_argument("--RKorder", type=int, default=0)
parser.add_argument("--output_folder",default="OUTPUT")
parser.add_argument("--formulation", default='BA')
parser.add_argument("--icase", default='x')
args = parser.parse_args()

global_stuff.icase=args.icase

if args.output_folder != 'OUTPUT':
   output_folder=str(args.output_folder)
   print('output_folder=',output_folder)

if args.e >= 0:
    experiment = args.e
print("experiment=", experiment)

match experiment:
    case 0:
        from experiment0 import *
    case 1:
        from experiment1 import *
    case 2:
        from experiment2 import *
    case 3:
        from experiment3 import *
    case 4:
        from experiment4 import *
    case 5:
        from experiment5 import *
    case 6:
        from experiment6 import *
    case 7:
        from experiment7 import *
    case 8:
        from experiment8 import *
    case 9:
        from experiment9 import *
    case 10:
        from experiment10 import *
    case 11:
        from experiment11 import *
    case 12:
        from experiment12 import *
    case 13:
        from experiment13 import *
    case 14:
        from experiment14 import *
    case 15:
        from experiment15 import *
    case 16:
        from experiment16 import *
    case 17:
        from experiment17 import *
    case 18:
        from experiment18 import *
    case 19:
        from experiment19 import *
    case 20:
        from experiment20 import *
    case 21:
        from experiment21 import *
    case 22:
        from experiment22 import *
    case 23:
        from experiment23 import *
    case 24:
        from experiment24 import *
    case 25:
        from experiment25 import *
    case 26:
        from experiment26 import *
    case 27:
        from experiment27 import *
    case 28:
        from experiment28 import *
    case _:
        exit("setup - unknown experiment")

if args.nelx > 0:
    nelx = args.nelx
print("nelx=", nelx)

if args.nelz > 0:
    nelz = args.nelz
print("nelz=", nelz)

if args.nstep > 0:
    nstep = args.nstep
print("nstep=", nstep)

if args.nq_per_dim > 0:
    nq_per_dim = args.nq_per_dim
print("nq_per_dim=", nq_per_dim)

if args.axisymmetric == 1:
    axisymmetric = True
print("axisymmetric=", axisymmetric)

if args.straighten_edges == 1:
    straighten_edges = True
print("straighten_edges=", straighten_edges)

if args.remove_rho_profile == 1:
    remove_rho_profile = True
print("remove_rho_profile", remove_rho_profile)

if args.particle_distribution > 0:
    particle_distribution = args.particle_distribution
print("particle_distribution=", particle_distribution)

if args.nparticle_per_dim > 0:
    nparticle_per_dim = args.nparticle_per_dim
print("nparticle_per_dim=", nparticle_per_dim)

if args.averaging == 1:
    averaging = "arithmetic"
if args.averaging == 2:
    averaging = "geometric"
if args.averaging == 3:
    averaging = "harmonic"
print("averaging=", averaging)

if args.nodal_projection_type > 0:
    nodal_projection_type = args.nodal_projection_type
print("nodal_projection_type=", args.nodal_projection_type)

if args.particle_rho_projection == 1:
    particle_rho_projection = "elemental"
if args.particle_rho_projection == 2:
    particle_rho_projection = "nodal"
if args.particle_rho_projection == 3:
    particle_rho_projection = "least_squares_P1"
if args.particle_rho_projection == 4:
    particle_rho_projection = "least_squares_Q1"
print("particle_rho_projection=", particle_rho_projection)

if args.particle_eta_projection == 1:
    particle_eta_projection = "elemental"
if args.particle_eta_projection == 2:
    particle_eta_projection = "nodal"
if args.particle_eta_projection == 3:
    particle_eta_projection = "least_squares_P1"
if args.particle_eta_projection == 4:
    particle_eta_projection = "least_squares_Q1"
print("particle_eta_projection=", particle_eta_projection)

if args.RKorder > 0:
    RKorder = args.RKorder
print("RKorder=", args.RKorder)

formulation=str(args.formulation)


try:
    os.mkdir(output_folder)
    print(f"Directory '{output_folder}' created successfully.")
except FileExistsError:
    print(f"Directory '{output_folder}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{output_folder}'.")
except Exception as e:
    print(f"An error occurred: {e}")

try:
    os.mkdir(output_folder+'/bottom')
    print(f"Directory '{output_folder}/bottom' created successfully.")
except FileExistsError:
    print(f"Directory '{output_folder}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{output_folder}'.")
except Exception as e:
    print(f"An error occurred: {e}")

try:
    os.mkdir(output_folder+'/top')
    print(f"Directory '{output_folder}' created successfully.")
except FileExistsError:
    print(f"Directory '{output_folder}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{output_folder}'.")
except Exception as e:
    print(f"An error occurred: {e}")

try:
    os.mkdir(output_folder+'/SWARM')
    print(f"Directory '{output_folder}/SWARM' created successfully.")
except FileExistsError:
    print(f"Directory '{output_folder}/SWARM' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{output_folder}/SWARM'.")
except Exception as e:
    print(f"An error occurred: {e}")

try:
    os.mkdir(output_folder+'/profiles')
    print(f"Directory '{output_folder}/profiles' created successfully.")
except FileExistsError:
    print(f"Directory '{output_folder}/profiles' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{output_folder}/profiles'.")
except Exception as e:
    print(f"An error occurred: {e}")



###############################################################################

match geometry:
    case "box":
        L_ref = (Lx + Lz) / 2
        if m_V==5:
           nn_V = (nelx + 1) * (nelz + 1) +nelx*nelz # number of V nodes
        else:
           nn_V = (2 * nelx + 1) * (2 * nelz + 1)  # number of V nodes
        nn_P = (nelx + 1) * (nelz + 1)  # number of P nodes
        opening_angle = 0
        hrad = 0
        htheta = 0
        theta_min = 0 
    case "eighth":
        nn_V = (2 * nelx + 1) * (2 * nelz + 1)  # number of V nodes
        nn_P = (nelx + 1) * (nelz + 1)  # number of P nodes
        opening_angle = np.pi / 4
        theta_min = np.pi / 4
        hrad = (Router - Rinner) / nelz
        htheta = opening_angle / nelx
        L_ref = (Rinner + Router) / 2
        Lx = 1
        Lz = 1
    case "quarter":
        nn_V = (2 * nelx + 1) * (2 * nelz + 1)  # number of V nodes
        nn_P = (nelx + 1) * (nelz + 1)  # number of P nodes
        opening_angle = np.pi / 2
        theta_min = 0
        hrad = (Router - Rinner) / nelz
        htheta = opening_angle / nelx
        L_ref = (Rinner + Router) / 2
        Lx = 1
        Lz = 1
    case "half":
        nn_V = (2 * nelx + 1) * (2 * nelz + 1)  # number of V nodes
        nn_P = (nelx + 1) * (nelz + 1)  # number of P nodes
        opening_angle = np.pi
        theta_min = -np.pi / 2
        hrad = (Router - Rinner) / nelz
        htheta = opening_angle / nelx
        L_ref = (Rinner + Router) / 2
        Lx = 1
        Lz = 1
    case "annulus":
        nn_V = (2 * nelx) * (2 * nelz + 1)  # number of V nodes
        nn_P = nelx * (nelz + 1)  # number of P nodes
        opening_angle = 2 * np.pi
        theta_min = 0
        hrad = (Router - Rinner) / nelz
        htheta = opening_angle / nelx
        L_ref = (Rinner + Router) / 2
        Lx = 1
        Lz = 1
    case _:
        exit("unknown geometry")

ndof_V = 2  # number of velocity dofs per node
nel = nelx * nelz  # total number of elements

if m_V==5:
   nn_T=nn_P # i.e. =4
else:
   nn_T=nn_V # i.e. =9

ndof_V_el = m_V * ndof_V

Nfem_V = nn_V * ndof_V  # number of velocity dofs
Nfem_P = nn_P  # number of pressure dofs
Nfem_T = nn_T  # number of temperature dofs
Nfem = Nfem_V + Nfem_P  # total nb of dofs

nparticle_per_element = nparticle_per_dim**2
nparticle = nel * nparticle_per_element

timings = np.zeros(39 + 1)
timings_mem = np.zeros(39 + 1)

blocks = False  # TODO change name & fix !

###############################################################################
# @@ quadrature rule points and weights
###############################################################################

qcoords, qweights, nq_per_element, nq = quadrature_setup(nq_per_dim, nel)

###############################################################################
# @@ open output files & write headers
###############################################################################

(
    vrms_file,
    pstats_file,
    vstats_file,
    srstats_file,
    dt_file,
    ptcl_stats_file,
    ptcl_active_file,
    timings_file,
    TM_file,
    EK_file,
    TVD_file,
    WAG_file,
    T_avrg_file,
    eta_avrg_file,
    delta_file,
    pvd_solution_file,
    pvd_swarm_file,
    etaq_file,
    etan_file,
    etae_file,
    corner_q_file,
    Tstats_file,
    Nu_file,
    avrg_T_bot_file,
    avrg_T_top_file,
    avrg_dTdz_bot_file,
    avrg_dTdz_top_file,
    bc_vel_file,
    conv_file,
) = open_files(vel_unit, time_unit, output_folder)

###############################################################################

volume = compute_domain_volume(geometry, Lx, Lz, Rinner, Router)

if nstep == 1:
    CFLnb = 0
print("axisymmetric=", axisymmetric)
print("geometry=", geometry)
print("straighten_edges=", straighten_edges)
print("remove_rho_profile=", remove_rho_profile)
print("nelx,nelz=", nelx, nelz)
print("Lx,Lz=", Lx, Lz)
print("nn_V=", nn_V, "| nn_P=", nn_P, "| nel=", nel)
print("Nfem_V=", Nfem_V, "| Nfem_P=", Nfem_P, "| Nfem=", Nfem)
print("nq_per_dim=", nq_per_dim)
print("CFLnb=", CFLnb)
print("debug_ascii:", debug_ascii, "| debug_nan:", debug_nan)
print("solve_T:", solve_T)
print("end_time=", end_time / time_scale, time_unit)
print("averaging:", averaging)
print("formulation:", formulation)
print("particle_distribution=", particle_distribution)
print("RKorder=", RKorder)
print("nparticle_per_dim=", nparticle_per_dim)
print("nparticle=", nparticle)
print("every_solution_vtu", every_solution_vtu)
print("every_swarm_vtu", every_swarm_vtu)
print("every_quadpoints_vtu", every_quadpoints_vtu)
print("rho_DT_top", rho_DT_top)
print("rho_DT_bot", rho_DT_bot)
print("gravity_npts=", gravity_npts)
print("top_free_slip=", top_free_slip, "| bot_free_slip=", bot_free_slip)
if geometry == "quarter" or geometry == "half" or geometry == "eighth" or geometry == "annulus":
    print("Rinner,Router=", Rinner, Router)
    print("hrad=", hrad)
print("======================================================")

###############################################################################
# if RKorder==-1 I hijack the particles and place them on the quadrature
# points so that there is no projection of the particles onto the
# quadrature points to speak of.

if RKorder == -1:
    nparticle_per_dim = nq_per_dim
    nparticle_per_element = nq_per_element
    nparticle = nq
    particle_distribution = -1
    particle_rho_projection = "qpts"
    particle_eta_projection = "qpts"

###############################################################################
# @@ build velocity nodes coordinates
###############################################################################
start = clock.time()

(
    hx,
    hz,
    nnx,
    nnz,
    x_V,
    z_V,
    rad_V,
    theta_V,
    top_Vnodes,
    bot_Vnodes,
    left_Vnodes,
    right_Vnodes,
    middleH_nodes,
    middleV_nodes,
    hull_Vnodes,
    cornerBL,
    cornerBR,
    cornerTL,
    cornerTR,
) = build_velocity_mesh(geometry, m_V, nn_V, nelx, nelz, Lx, Lz, Rinner, Router, opening_angle, debug_ascii)

print("build V mesh: ................................ %.3f s" % (clock.time() - start))

###############################################################################
# @@ connectivity for velocity nodes
###############################################################################
start = clock.time()

icon_V, top_element, bot_element, left_element, right_element, middleH_element, middleV_element = (
    build_velocity_connectivity(geometry, m_V, nelx, nelz, nnx, nnz, middleH_nodes, middleV_nodes)
)

print("build icon_V: ................................ %.3f s" % (clock.time() - start))

###############################################################################
# in the case of a curved axisymmetric domain, it could be beneficial to
# straighten the element sides
###############################################################################

x_V, z_V = straighten_edges_axisymmetric(geometry, axisymmetric, straighten_edges, nel, icon_V, x_V, z_V)

###############################################################################
# @@ build pressure grid
###############################################################################
start = clock.time()

x_P, z_P, rad_P, theta_P, left_Pnodes, right_Pnodes, top_Pnodes, bot_Pnodes, hull_Pnodes = build_pressure_mesh(
    geometry, nn_P, nelx, nelz, hx, hz, Rinner, Router, opening_angle, debug_ascii
)

print("build P grid: ................................ %.3f s" % (clock.time() - start))

###############################################################################
# @@ build pressure connectivity array
###############################################################################
start = clock.time()

icon_P = build_pressure_connectivity(geometry, nelx, nelz, nel, m_P)

print("build icon_P: ................................ %.3f s" % (clock.time() - start))

###############################################################################
# @@ build temperature grid
# if quad-mini elements are used (m_V=5) then the temperature FE space is Q1
# so that the T nodes and connectivity is the same as pressure. 
# if Q2Q1 elements are used (m_V=9) then the temperature FE space is Q2 also.
###############################################################################
start = clock.time()

if m_V==5:
   x_T=np.copy(x_P)
   z_T=np.copy(z_P)
   rad_T=np.copy(rad_P)
   theta_T=np.copy(theta_P)
   top_Tnodes=np.copy(top_Pnodes)
   bot_Tnodes=np.copy(bot_Pnodes)
   left_Tnodes=np.copy(left_Pnodes)
   right_Tnodes=np.copy(right_Pnodes)
   hull_Tnodes=np.copy(hull_Pnodes)   # does not exist!
else:
   x_T=np.copy(x_V)
   z_T=np.copy(z_V)
   rad_T=np.copy(rad_V)
   theta_T=np.copy(theta_V)
   top_Tnodes=np.copy(top_Vnodes)
   bot_Tnodes=np.copy(bot_Vnodes)
   left_Tnodes=np.copy(left_Vnodes)
   right_Tnodes=np.copy(right_Vnodes)
   hull_Tnodes=np.copy(hull_Vnodes)

print("build T grid: ................................ %.3f s" % (clock.time() - start))

###############################################################################
# @@ build temperature connectivity array
###############################################################################
start = clock.time()

if m_V==5:
   icon_T=np.copy(icon_P)
else:
   icon_T=np.copy(icon_V)

print("build icon_T: ................................ %.3f s" % (clock.time() - start))

###############################################################################
# @@ apply mesh stretching 
###############################################################################
start = clock.time()

if use_stretching_x:
   x_V,x_P,x_T,xi=apply_mesh_stretching_x(m_V,x_V,x_P,nelx,nelz,icon_V,icon_P,x_segments,nelx_segments,Lx)
   np.savetxt("DEBUG/mesh_after_x_stretching.ascii", np.array([x_V, z_V]).T)

if use_stretching_z:
   z_V,z_P,z_T,zeta=apply_mesh_stretching_z(m_V,z_V,z_P,nelx,nelz,icon_V,icon_P,z_segments,nelz_segments,Lz)
   np.savetxt("DEBUG/mesh_after_z_stretching.ascii", np.array([x_V, z_V]).T)

if debug_ascii: output_mesh_V(x_V,z_V,icon_V,nn_V,m_V,nel)

print("apply mesh stretching: ....................... %.3f s" % (clock.time() - start))

###############################################################################
# @@ prescribe initial topography if necessary
###############################################################################
start = clock.time()

if use_free_surface:
   z_V,z_P,z_T=\
   prescribe_initial_topography(experiment,Lx,Lz,nn_V,nelx,nelz,x_V,z_V,z_P,z_T,\
                                icon_V,icon_P,top_Vnodes,m_V,use_stretching)

print("prescrive initial topo: ...................... %.3f s" % (clock.time() - start))

###############################################################################
# @@ define velocity boundary conditions
###############################################################################
start = clock.time()

bc_fix_V, bc_val_V = assign_boundary_conditions_V(
    x_V,
    z_V,
    rad_V,
    theta_V,
    ndof_V,
    Nfem_V,
    nn_V,
    hull_Vnodes,
    top_Vnodes,
    bot_Vnodes,
    left_Vnodes,
    right_Vnodes,
)

print("velocity b.c.: ............................... %.3f s" % (clock.time() - start))

###############################################################################
# @@ define temperature boundary conditions
###############################################################################
start = clock.time()

if solve_T:
    bc_fix_T, bc_val_T = assign_boundary_conditions_T(
        x_T,
        z_T,
        rad_T,
        theta_T,
        Nfem_T,
        nn_T,
        hull_Tnodes,
        top_Tnodes,
        bot_Tnodes,
        left_Tnodes,
        right_Tnodes,
    )
else:
    bc_fix_T = False

print("temperature b.c.: ............................ %.3f s" % (clock.time() - start))

###############################################################################
# @@ initial temperature
###############################################################################
start = clock.time()

if solve_T:
    T = initial_temperature(x_T, z_T, rad_T, theta_T, nn_T)

    for i in range(nn_T):
        if bc_fix_T[i]:
            T[i] = bc_val_T[i]

    T_mem = T.copy()

    if debug_ascii:
        np.savetxt("DEBUG/T_init.ascii", np.array([x_T, z_T, T]).T, header="# x,z,T")

    print("     -> T init (m,M) %.3e %.3e " % (np.min(T) - TKelvin, np.max(T) - TKelvin))

    print("initial temperature: ......................... %.3f s" % (clock.time() - start))

else:
    T = np.zeros(nn_T, dtype=np.float64)
    T_mem = T.copy()

###############################################################################
# @@ define_mapping
###############################################################################

# x_M,z_M,m_M=define_mapping(geometry,mapping,nelx,nel,x_V,z_V,icon_V,rad_V,theta_V)
# if debug_ascii: i
# np.savetxt('DEBUG/mesh_M.ascii',np.array([x_M.flatten(),z_M.flatten()]).T,header='# x,z')
# exit()

###############################################################################
# @@ compute area of elements / sanity check
# @@ precompute basis functions values at quadrature points
###############################################################################
start = clock.time()

(
    rq,
    tq,
    weightq,
    N_V,
    N_P,
    N_T,
    dNdr_V,
    dNdt_V,
    dNdr_P,
    dNdt_P,
    dNdr_T,
    dNdt_T,
    JxWq,
    jcbi00q,
    jcbi01q,
    jcbi10q,
    jcbi11q,
    area,
) = basis_functions_setup_q(nq_per_dim, m_V, m_P, m_T, nel, x_V, z_V, icon_V, qcoords, qweights, volume)

print("comp N, grad(N), elts areas at q pts: ........ %.3f s" % (clock.time() - start))

###############################################################################
# @@ compute center coordinates
###############################################################################
start = clock.time()

x_e, z_e, rad_e, theta_e = compute_center_coordinates(geometry, nel, x_V, z_V, icon_V)

print("compute center coordinates: .................. %.3f s" % (clock.time() - start))

###############################################################################
# @@ precompute basis functions and jacobian values at V nodes
###############################################################################
start = clock.time()

N_P_n, dNdr_V_n, dNdt_V_n, jcbi00n, jcbi01n, jcbi10n, jcbi11n = basis_functions_setup_Vnodes(
    m_V, m_P, nel, r_V, t_V, x_V, z_V, icon_V
)

print("compute N_P & grad(N_V) at V nodes: .......... %.3f s" % (clock.time() - start))

###############################################################################
# @@ precompute basis functions and jacobian values at T nodes
###############################################################################
start = clock.time()

dNdr_T_n, dNdt_T_n, jcbi00_T, jcbi01_T, jcbi10_T, jcbi11_T = basis_functions_setup_Tnodes(
    m_T, nel, r_T, t_T, x_T, z_T, icon_T
)

print("compute grad(N_T) at V nodes: ................ %.3f s" % (clock.time() - start))

###############################################################################
# @@ compute coordinates of quadrature points
# xq,zq are size (nel,nq_per_element)
###############################################################################
start = clock.time()

xq = project_nodal_Vfield_onto_qpoints(x_V, nq_per_element, nel, m_V, N_V, icon_V)
zq = project_nodal_Vfield_onto_qpoints(z_V, nq_per_element, nel, m_V, N_V, icon_V)

print("     -> xq (m,M) %.3e %.3e " % (np.min(xq), np.max(xq)))
print("     -> zq (m,M) %.3e %.3e " % (np.min(zq), np.max(zq)))

if debug_ascii:
    np.savetxt("DEBUG/qpoints.ascii", np.array([xq.flatten(), zq.flatten()]).T, header="# x,z")

print("compute coords quad pts: ..................... %.3f s" % (clock.time() - start))

###############################################################################
# @@ compute gravity vector at quadrature points
# gx_q,gz_q are size (nel,nq_per_element)
###############################################################################
start = clock.time()

gx_q = np.zeros((nel, nq_per_element), dtype=np.float64)
gz_q = np.zeros((nel, nq_per_element), dtype=np.float64)

for iel in range(0, nel):
    for iq in range(0, nq_per_element):
        gx_q[iel, iq], gz_q[iel, iq] = gravity_model(xq[iel, iq], zq[iel, iq])

print("     -> gx_q (m,M) %.3e %.3e " % (np.min(gx_q), np.max(gx_q)))
print("     -> gz_q (m,M) %.3e %.3e " % (np.min(gz_q), np.max(gz_q)))

if debug_ascii:
    np.savetxt(
        "DEBUG/qgravity.ascii",
        np.array([xq.flatten(), zq.flatten(), gx_q.flatten(), gz_q.flatten()]).T,
        header="#x,z,gx,gz",
    )

print("assign qpts gravity vector: .................. %.3f s" % (clock.time() - start))

###############################################################################
# @@ compute gravity on mesh points
###############################################################################
start = clock.time()

gx_n = np.zeros(nn_V, dtype=np.float64)
gz_n = np.zeros(nn_V, dtype=np.float64)
gx_e = np.zeros(nel, dtype=np.float64)
gz_e = np.zeros(nel, dtype=np.float64)

for i in range(0, nn_V):
    gx_n[i], gz_n[i] = gravity_model(x_V[i], z_V[i])

gr_n = gx_n * np.cos(theta_V) + gz_n * np.sin(theta_V)

for iel in range(0, nel):
    gx_e[iel], gz_e[iel] = gravity_model(x_e[iel], z_e[iel])

gr_e = gx_e * np.cos(theta_e) + gz_e * np.sin(theta_e)

if debug_ascii:
    np.savetxt("DEBUG/gr_n.ascii", np.array([x_V, z_V, gr_n]).T, header="#x,z,gr")
    np.savetxt("DEBUG/gr_e.ascii", np.array([x_e, z_e, gr_e]).T, header="#x,z,gr")

print("compute grav on nodes: ....................... %.3f s" % (clock.time() - start))

###############################################################################
# @@ compute normal vector of domain - NOT needed anymore
###############################################################################
# start=clock.time()
# nx,nz=compute_normals(geometry,nel,nn_V,nq_per_element,m_V,icon_V,dNdr_V,dNdt_V,\
#                      JxWq,hull_nodes,jcbi00q,jcbi01q,jcbi10q,jcbi11q)
# if debug_ascii: np.savetxt('DEBUG/normal_vector.ascii',np.array([x_V[hull_nodes],\
#                           z_V[hull_nodes],nx[hull_nodes],nz[hull_nodes]]).T,header='#x,z,nx,nz')
#
# print("compute normal vector: %.3f s" % (clock.time()-start))

###############################################################################
# @@ compute array for assembly
###############################################################################
start = clock.time()

local_to_globalV = np.zeros((ndof_V_el, nel), dtype=np.int32)

for iel in range(0, nel):
    for k1 in range(0, m_V):
        for i1 in range(0, ndof_V):
            ikk = ndof_V * k1 + i1
            m1 = ndof_V * icon_V[k1, iel] + i1
            local_to_globalV[ikk, iel] = m1

print("compute local_to_globalV: .................... %.3f s" % (clock.time() - start))

###############################################################################
# @@ fill II_Stokes,JJ_Stokes arrays for Stokes matrix
###############################################################################
start = clock.time()

if solve_Stokes:
    bignb_Stokes = nel * (ndof_V_el**2 + 2 * ndof_V_el * m_P)

    II_Stokes = np.zeros(bignb_Stokes, dtype=np.int32)
    JJ_Stokes = np.zeros(bignb_Stokes, dtype=np.int32)

    counter = 0
    for iel in range(0, nel):
        for ikk in range(ndof_V_el):
            m1 = local_to_globalV[ikk, iel]
            for jkk in range(ndof_V_el):
                m2 = local_to_globalV[jkk, iel]
                II_Stokes[counter] = m1
                JJ_Stokes[counter] = m2
                counter += 1
            for jkk in range(0, m_P):
                m2 = icon_P[jkk, iel] + Nfem_V
                II_Stokes[counter] = m1
                JJ_Stokes[counter] = m2
                counter += 1
                II_Stokes[counter] = m2
                JJ_Stokes[counter] = m1
                counter += 1

print("fill II_Stokes,JJ_Stokes arrays: ............. %.3f s" % (clock.time() - start))

###############################################################################
# @@ fill II_K,JJ_K arrays for K block matrix (Q2, vector)
###############################################################################
start = clock.time()

if solve_Stokes:
    bignb_K = nel * ndof_V_el**2

    II_K = np.zeros(bignb_K, dtype=np.int32)
    JJ_K = np.zeros(bignb_K, dtype=np.int32)

    counter = 0
    for iel in range(0, nel):
        for ikk in range(ndof_V_el):
            m1 = local_to_globalV[ikk, iel]
            for jkk in range(ndof_V_el):
                m2 = local_to_globalV[jkk, iel]
                II_K[counter] = m1
                JJ_K[counter] = m2
                counter += 1

print("fill II_K,JJ_K arrays: ....................... %.3f s" % (clock.time() - start))

###############################################################################
# @@ fill II_K,JJ_K arrays for G and GT block matrices
###############################################################################
start = clock.time()

if solve_Stokes:
    bignb_G = nel * (m_P * ndof_V_el)

    II_G = np.zeros(bignb_G, dtype=np.int32)
    JJ_G = np.zeros(bignb_G, dtype=np.int32)
    II_GT = np.zeros(bignb_G, dtype=np.int32)
    JJ_GT = np.zeros(bignb_G, dtype=np.int32)

    counter = 0
    for iel in range(0, nel):
        for ikk in range(ndof_V_el):
            m1 = local_to_globalV[ikk, iel]
            for jkk in range(0, m_P):
                m2 = icon_P[jkk, iel]
                II_G[counter] = m1
                JJ_G[counter] = m2
                counter += 1

    counter = 0
    for iel in range(0, nel):
        for jkk in range(0, m_P):
            m1 = icon_P[jkk, iel]
            for ikk in range(ndof_V_el):
                m2 = local_to_globalV[ikk, iel]
                II_GT[counter] = m1
                JJ_GT[counter] = m2
                counter += 1

print("fill II_G,JJ_G arrays: ....................... %.3f s" % (clock.time() - start))

###############################################################################
# @@ fill II_MP,JJ_P arrays for pressure mass matrix (Q1, scalar)
###############################################################################
start = clock.time()

if solve_Stokes:
    bignb_P = nel * m_P**2

    II_MP = np.zeros(bignb_P, dtype=np.int32)
    JJ_MP = np.zeros(bignb_P, dtype=np.int32)

    counter = 0
    for iel in range(0, nel):
        for ikk in range(m_P):
            m1 = icon_P[ikk, iel]
            for jkk in range(m_P):
                m2 = icon_P[jkk, iel]
                II_MP[counter] = m1
                JJ_MP[counter] = m2
                counter += 1

print("fill II_MP,JJ_MP arrays: ..................... %.3f s" % (clock.time() - start))

###############################################################################
# @@ fill II_T,JJ_T arrays for temperature matrix & plith matrix (Q2, scalar)
###############################################################################
start = clock.time()

bignb_T = nel * m_T**2

II_T = np.zeros(bignb_T, dtype=np.int32)
JJ_T = np.zeros(bignb_T, dtype=np.int32)

counter = 0
for iel in range(0, nel):
    for ikk in range(m_T):
        m1 = icon_T[ikk, iel]
        for jkk in range(m_T):
            m2 = icon_T[jkk, iel]
            II_T[counter] = m1
            JJ_T[counter] = m2
            counter += 1

print("fill II_T,JJ_T arrays: ....................... %.3f s" % (clock.time() - start))

###############################################################################
# @@ particle coordinates setup
###############################################################################
start = clock.time()

swarm_active,swarm_x,swarm_z,swarm_r,swarm_t,swarm_id,swarm_iel, swarm_rad, swarm_theta,swarm_paint=\
swarm_coordinates_setup(geometry,particle_distribution,nparticle,nparticle_per_element,nparticle_per_dim,\
nel,nq_per_dim,nelx,nelz,Lx,Lz,hx,hz,xq,zq,qcoords,x_V,z_V,icon_V,Rinner,Router,opening_angle,theta_min)

if debug_ascii:
    np.savetxt("DEBUG/swarm_distribution.ascii", np.array([swarm_x, swarm_z]).T, header="#x,z")

print("     -> nparticle %d " % nparticle)
print("     -> swarm_x (m,M) %.3e %.3e " % (np.min(swarm_x), np.max(swarm_x)))
print("     -> swarm_z (m,M) %.3e %.3e " % (np.min(swarm_z), np.max(swarm_z)))
print("     -> swarm_r (m,M) %.3e %.3e " % (np.min(swarm_r), np.max(swarm_r)))
print("     -> swarm_t (m,M) %.3e %.3e " % (np.min(swarm_t), np.max(swarm_t)))
print("     -> swarm_id (m,M) %.3e %.3e " % (np.min(swarm_id), np.max(swarm_id)))
print("     -> swarm_iel (m,M) %.3e %.3e " % (np.min(swarm_iel), np.max(swarm_iel)))

print("particles setup: ............................. %.3f s" % (clock.time() - start))

###############################################################################
# @@ initial strain setup
# TODO: I should late implement a generic function to allow initial strain weakening/weak seeds
###############################################################################

swarm_strain = np.zeros(nparticle, dtype=np.float64)

###############################################################################
# @@ particle layout
###############################################################################
start = clock.time()

swarm_wf, material_names = particle_layout(nparticle, nmat, swarm_x, swarm_z, swarm_rad, swarm_theta, Lx, Lz)

for imat in range(0, nmat):
    print(
        "     -> swarm_weight_fraction of mat %d (m,M) %d %d "
        % (imat, np.min(swarm_wf[imat, :]), np.max(swarm_wf[imat, :]))
    )

if debug_ascii:
    for imat in range(0, nmat):
        np.savetxt(
            "DEBUG/swarm_material" + str(imat) + ".ascii",
            np.array([swarm_x, swarm_z, swarm_wf[imat, :]]).T,
            header="#x,z,mat",
        )

if use_melting:
    swarm_F = np.zeros(nparticle, dtype=np.float32)
    swarm_sst = np.zeros(nparticle, dtype=np.float32)
else:
    swarm_F = 0.0
    swarm_sst = 0.0

print("particle layout: ............................. %.3f s" % (clock.time() - start))

###############################################################################
###############################################################################
###############################################################################
# @@ --------------------- time stepping loop ----------------------------------
###############################################################################
###############################################################################
###############################################################################

geo_time = 0.0
dt1_mem = 1e50
dt2_mem = 1e50

exx_n = np.zeros(nn_V, dtype=np.float64)
ezz_n = np.zeros(nn_V, dtype=np.float64)
exz_n = np.zeros(nn_V, dtype=np.float64)
dpdx_n = np.zeros(nn_V, dtype=np.float64)
dpdz_n = np.zeros(nn_V, dtype=np.float64)
u_mem = np.zeros(nn_V, dtype=np.float64)
w_mem = np.zeros(nn_V, dtype=np.float64)
p_mem = np.zeros(nn_P, dtype=np.float64)
q = np.zeros(nn_V, dtype=np.float64)

topstart = clock.time()

iter_nl=0
istep=0

for iloop in range(0, nstep*niter_nl):

    print("======================================================")
    print("istep= %d | iter_nl= %d | time= %.4e | iloop= %d" % (istep, iter_nl, geo_time / time_scale,iloop))
    print("======================================================")
       
    inside_nonlinear_iterations=False

    ###############################################################################################
    # @@ interpolate strain rate, pressure and temperature on particles
    ###############################################################################################
    start = clock.time()

    swarm_exx = interpolate_field_on_particles(nparticle, swarm_active, swarm_r, swarm_t, swarm_iel, exx_n, icon_V)
    swarm_ezz = interpolate_field_on_particles(nparticle, swarm_active, swarm_r, swarm_t, swarm_iel, ezz_n, icon_V)
    swarm_exz = interpolate_field_on_particles(nparticle, swarm_active, swarm_r, swarm_t, swarm_iel, exz_n, icon_V)

    swarm_p = interpolate_field_on_particles(nparticle, swarm_active, swarm_r, swarm_t, swarm_iel, q, icon_V)

    if solve_T:
        swarm_T = interpolate_field_on_particles(nparticle, swarm_active, swarm_r, swarm_t, swarm_iel, T, icon_T)
        print("     -> swarm_T (m,M) %.3e %.3e " % (np.min(swarm_T) - TKelvin, np.max(swarm_T) - TKelvin))
    else:
        swarm_T = 0

    print("interp sr, q, T on particles: ................ %.3f s" % (clock.time() - start))
    timings[24] += clock.time() - start

    ###############################################################################################
    # @@ compute depletion and super solidus temperature
    ###############################################################################################
    start = clock.time()

    if istep > 0 and use_melting and solve_T:
        swarm_F, swarm_sst = update_F(nparticle, swarm_active,swarm_p, swarm_T, swarm_F)

        print("****************************************")

        print("     -> swarm_F (m,M) %.3e %.3e " % (np.min(swarm_F), np.max(swarm_F)))
        print("     -> swarm_sst (m,M) %.3e %.3e " % (np.min(swarm_sst), np.max(swarm_sst)))

        print("melting on particles: ......................... %.3f s" % (clock.time() - start))

    ###############################################################################################
    # @@ evaluate density and viscosity on particles (and hcond, hcapa, hprod)
    # if solve_T is false then swarm_{hcond,hcapa,hprod} are scalars equal to zero
    ###############################################################################################
    start = clock.time()

    swarm_rho, swarm_eta, swarm_hcond, swarm_hcapa, swarm_hprod, swarm_alpha, swarm_mechanism = material_model(
        nparticle,
        swarm_active,
        nmat,
        swarm_wf,
        swarm_x,
        swarm_z,
        swarm_rad,
        swarm_theta,
        swarm_exx,
        swarm_ezz,
        swarm_exz,
        swarm_T,
        swarm_p,
    )

    if verbose_output:
       print("     -> swarm_rho (m,M) %.5e %.5e " % (np.min(swarm_rho[swarm_active]), np.max(swarm_rho[swarm_active])))
       print("     -> swarm_eta (m,M) %.5e %.5e " % (np.min(swarm_eta[swarm_active]), np.max(swarm_eta[swarm_active])))

    if solve_T and verbose_output:
        print("     -> swarm_hcapa (m,M) %.4e %.4e " % (np.min(swarm_hcapa[swarm_active]), np.max(swarm_hcapa[swarm_active])))
        print("     -> swarm_hcond (m,M) %.4e %.4e " % (np.min(swarm_hcond[swarm_active]), np.max(swarm_hcond[swarm_active])))
        print("     -> swarm_hprod (m,M) %.4e %.4e " % (np.min(swarm_hprod[swarm_active]), np.max(swarm_hprod[swarm_active])))
        print("     -> swarm_alpha (m,M) %.4e %.4e " % (np.min(swarm_alpha[swarm_active]), np.max(swarm_alpha[swarm_active])))

    if debug_ascii:
        np.savetxt("DEBUG/swarm_rho.ascii", np.array([swarm_x[swarm_active], swarm_z[swarm_active], swarm_rho[swarm_active]]).T, header="# x,z,rho")
    if debug_ascii:
        np.savetxt("DEBUG/swarm_eta.ascii", np.array([swarm_x[swarm_active], swarm_z[swarm_active], swarm_eta[swarm_active]]).T, header="# x,z,eta")

    print("call material model on particles: ............ %.3f s" % (clock.time() - start))
    timings[15] += clock.time() - start

    ###############################################################################################
    # @@ project particle properties on elements
    # this is also where the nparticle_e array is filled
    ###############################################################################################
    start = clock.time()

    rho_e, eta_e, nparticle_e = project_particles_on_elements(
        nel, nparticle, swarm_active, swarm_rho, swarm_eta, swarm_iel, averaging
    )

    if np.min(nparticle_e) == 0:
        exit("ABORT: an element contains no particle!")

    ptcl_stats_file.write("%.4e %d %d\n" % (geo_time / time_scale, np.min(nparticle_e), np.max(nparticle_e)))
    ptcl_stats_file.flush()
    etae_file.write("%.4e %.4e %.4e\n" % (geo_time / time_scale, np.min(eta_e), np.max(eta_e)))
    etae_file.flush()

    print("     -> rho_e (m,M) %.3e %.3e " % (np.min(rho_e), np.max(rho_e)))
    print("     -> eta_e (m,M) %.3e %.3e " % (np.min(eta_e), np.max(eta_e)))

    if debug_ascii:
        np.savetxt("DEBUG/rho_e.ascii", np.array([x_e, z_e, rho_e]).T, header="# x,z,rho")
        np.savetxt("DEBUG/eta_e.ascii", np.array([x_e, z_e, eta_e]).T, header="# x,z,eta")

    if debug_nan and np.isnan(np.sum(rho_e)):
        exit("nan found in rho_e")
    if debug_nan and np.isnan(np.sum(eta_e)):
        exit("nan found in eta_e")

    print("project particle fields on elements: ......... %.3f s" % (clock.time() - start))
    timings[17] += clock.time() - start

    ###############################################################################################
    # carry out least square fits of elemental density and viscosity
    ###############################################################################################
    start = clock.time()

    if particle_rho_projection == "least_squares_P1" or particle_eta_projection == "least_squares_P1":
        (
            ls_rho_a,
            ls_rho_b,
            ls_rho_c,
            ls_eta_a,
            ls_eta_b,
            ls_eta_c,
            rho_min_e,
            rho_max_e,
            eta_min_e,
            eta_max_e,
        ) = compute_ls_coefficients_P1(nel, x_e, z_e, swarm_x, swarm_z, swarm_iel, swarm_rho, swarm_eta)

        ls_rho_d = 0.0
        ls_eta_d = 0.0

        for iel in range(0, nel):
            ls_rho_b[iel], ls_rho_c[iel] = limiter(
                ls_rho_a[iel], ls_rho_b[iel], ls_rho_c[iel], rho_min_e[iel], rho_max_e[iel], hx
            )
            ls_eta_b[iel], ls_eta_c[iel] = limiter(
                ls_eta_a[iel], ls_eta_b[iel], ls_eta_c[iel], eta_min_e[iel], eta_max_e[iel], hx
            )

        print("     -> ls_rho_a (m,M) %.3e %.3e " % (np.min(ls_rho_a), np.max(ls_rho_a)))
        print("     -> ls_rho_b (m,M) %.3e %.3e " % (np.min(ls_rho_b), np.max(ls_rho_b)))
        print("     -> ls_rho_c (m,M) %.3e %.3e " % (np.min(ls_rho_c), np.max(ls_rho_c)))

        print("     -> ls_eta_a (m,M) %.3e %.3e " % (np.min(ls_eta_a), np.max(ls_eta_a)))
        print("     -> ls_eta_b (m,M) %.3e %.3e " % (np.min(ls_eta_b), np.max(ls_eta_b)))
        print("     -> ls_eta_c (m,M) %.3e %.3e " % (np.min(ls_eta_c), np.max(ls_eta_c)))

        output_fields_ls_P1(
            istep,
            nel,
            x_V,
            z_V,
            icon_V,
            x_e,
            z_e,
            ls_rho_a,
            ls_rho_b,
            ls_rho_c,
            ls_eta_a,
            ls_eta_b,
            ls_eta_c,
            output_folder
        )

    elif particle_rho_projection == "least_squares_Q1" or particle_eta_projection == "least_squares_Q1":
        (
            ls_rho_a,
            ls_rho_b,
            ls_rho_c,
            ls_rho_d,
            ls_eta_a,
            ls_eta_b,
            ls_eta_c,
            ls_eta_d,
            rho_min_e,
            rho_max_e,
            eta_min_e,
            eta_max_e,
        ) = compute_ls_coefficients_Q1(nel, x_e, z_e, swarm_x, swarm_z, swarm_iel, swarm_rho, swarm_eta)

        print("**** no limiter***")

        print("     -> ls_rho_a (m,M) %.3e %.3e " % (np.min(ls_rho_a), np.max(ls_rho_a)))
        print("     -> ls_rho_b (m,M) %.3e %.3e " % (np.min(ls_rho_b), np.max(ls_rho_b)))
        print("     -> ls_rho_c (m,M) %.3e %.3e " % (np.min(ls_rho_c), np.max(ls_rho_c)))
        print("     -> ls_rho_d (m,M) %.3e %.3e " % (np.min(ls_rho_d), np.max(ls_rho_d)))

        print("     -> ls_eta_a (m,M) %.3e %.3e " % (np.min(ls_eta_a), np.max(ls_eta_a)))
        print("     -> ls_eta_b (m,M) %.3e %.3e " % (np.min(ls_eta_b), np.max(ls_eta_b)))
        print("     -> ls_eta_c (m,M) %.3e %.3e " % (np.min(ls_eta_c), np.max(ls_eta_c)))
        print("     -> ls_eta_d (m,M) %.3e %.3e " % (np.min(ls_eta_d), np.max(ls_eta_d)))

        output_fields_ls_Q1(
            istep,
            nel,
            x_V,
            z_V,
            icon_V,
            x_e,
            z_e,
            ls_rho_a,
            ls_rho_b,
            ls_rho_c,
            ls_rho_d,
            ls_eta_a,
            ls_eta_b,
            ls_eta_c,
            ls_eta_d,
            output_folder,
        )

    else:
        ls_rho_a = 0.0
        ls_rho_b = 0.0
        ls_rho_c = 0.0
        ls_rho_d = 0.0
        ls_eta_a = 0.0
        ls_eta_b = 0.0
        ls_eta_c = 0.0
        ls_eta_d = 0.0

    print("least squares fit: ........................... %.3f s" % (clock.time() - start))
    timings[25] += clock.time() - start

    ###############################################################################################
    # @@ project particle properties on V nodes
    # nodal rho & eta are computed on nodes 0,1,2,3, while values on nodes
    # 4,5,6,7,8 are obtained by simple averages. In the end we obtaine Q1 fields
    # See pic.py for more details about the four flavours.
    ###############################################################################################
    start = clock.time()

    match nodal_projection_type:
        case 1:
            rho_n = project_particle_field_on_nodes_1(
                nel, nn_V, nparticle, swarm_active, swarm_rho, icon_V, swarm_iel, swarm_r, swarm_t, "arithmetic"
            )
            eta_n = project_particle_field_on_nodes_1(
                nel, nn_V, nparticle, swarm_active, swarm_eta, icon_V, swarm_iel, swarm_r, swarm_t, averaging
            )
        case 2:
            rho_n = project_particle_field_on_nodes_2(
                nel, nn_V, nparticle, swarm_active, swarm_rho, icon_V, swarm_iel, swarm_r, swarm_t, "arithmetic"
            )
            eta_n = project_particle_field_on_nodes_2(
                nel, nn_V, nparticle, swarm_active, swarm_eta, icon_V, swarm_iel, swarm_r, swarm_t, averaging
            )
        case 3:
            rho_n = project_particle_field_on_nodes_3(
                nel, nn_V, nparticle, swarm_active, swarm_rho, icon_V, swarm_iel, swarm_r, swarm_t, "arithmetic"
            )
            eta_n = project_particle_field_on_nodes_3(
                nel, nn_V, nparticle, swarm_active, swarm_eta, icon_V, swarm_iel, swarm_r, swarm_t, averaging
            )
        case 4:
            rho_n = project_particle_field_on_nodes_4(
                nel, nn_V, nparticle, swarm_active, swarm_rho, icon_V, swarm_iel, swarm_r, swarm_t, "arithmetic"
            )
            eta_n = project_particle_field_on_nodes_4(
                nel, nn_V, nparticle, swarm_active, swarm_eta, icon_V, swarm_iel, swarm_r, swarm_t, averaging
            )

    print("     -> rho_n (m,M) %.3e %.3e " % (np.min(rho_n), np.max(rho_n)))
    print("     -> eta_n (m,M) %.3e %.3e " % (np.min(eta_n), np.max(eta_n)))

    etan_file.write("%d %.3e %.3e\n" % (istep, np.min(eta_n), np.max(eta_n)))
    etan_file.flush()

    if debug_ascii:
        np.savetxt(
            "DEBUG/rho_n.ascii",
            np.array([x_V, z_V, rho_n, rad_V, theta_V]).T,
            header="# x,z,rho,rad,theta",
        )
    if debug_ascii:
        np.savetxt(
            "DEBUG/eta_n.ascii",
            np.array([x_V, z_V, eta_n, rad_V, theta_V]).T,
            header="# x,z,eta,rad,theta",
        )

    if solve_T:
        hcond_n = project_particle_field_on_nodes_2(
            nel, nn_V, nparticle, swarm_active, swarm_hcond, icon_V, swarm_iel, swarm_r, swarm_t, "arithmetic"
        )
        hcapa_n = project_particle_field_on_nodes_2(
            nel, nn_V, nparticle, swarm_active, swarm_hcapa, icon_V, swarm_iel, swarm_r, swarm_t, "arithmetic"
        )
        hprod_n = project_particle_field_on_nodes_2(
            nel, nn_V, nparticle, swarm_active, swarm_hprod, icon_V, swarm_iel, swarm_r, swarm_t, "arithmetic"
        )
        alpha_n = project_particle_field_on_nodes_2(
            nel, nn_V, nparticle, swarm_active, swarm_alpha, icon_V, swarm_iel, swarm_r, swarm_t, "arithmetic"
        )

        print("     -> hcond_n (m,M) %.3e %.3e " % (np.min(hcond_n), np.max(hcond_n)))
        print("     -> hcapa_n (m,M) %.3e %.3e " % (np.min(hcapa_n), np.max(hcapa_n)))
        print("     -> hprod_n (m,M) %.3e %.3e " % (np.min(hprod_n), np.max(hprod_n)))

        if debug_ascii:
            np.savetxt("DEBUG/hcond_n.ascii", np.array([x_V, z_V, hcond_n]).T, header="# x,z,hcond")
        if debug_ascii:
            np.savetxt("DEBUG/hcapa_n.ascii", np.array([x_V, z_V, hcapa_n]).T, header="# x,z,hcapa")
        if debug_ascii:
            np.savetxt("DEBUG/hprod_n.ascii", np.array([x_V, z_V, hprod_n]).T, header="# x,z,hprod")

    print("project particle fields on nodes: ............ %.3f s" % (clock.time() - start))
    timings[18] += clock.time() - start

    ###############################################################################################
    # compute (nodal) rho profile
    ###############################################################################################
    start = clock.time()

    coords_n_profile = np.zeros(nnz, dtype=np.float64)
    counter = 0
    for j in range(0, nnz):
        for i in range(0, nnx):
            if i == 0:
                if geometry == "box":
                    coords_n_profile[j] = z_V[counter]
                else:
                    coords_n_profile[j] = rad_V[counter]
            counter += 1

    coords_e_profile = np.zeros(nelz, dtype=np.float64)
    counter = 0
    for j in range(0, nelz):
        for i in range(0, nelx):
            if i == 0:
                if geometry == "box":
                    coords_e_profile[j] = z_e[counter]
                else:
                    coords_e_profile[j] = rad_e[counter]
            counter += 1

    rho_n_profile = np.zeros(nnz, dtype=np.float64)
    counter = 0
    for j in range(0, nnz):
        for i in range(0, nnx):
            rho_n_profile[j] += rho_n[counter]
            counter += 1
    rho_n_profile /= nnx

    rho_e_profile = np.zeros(nelz, dtype=np.float64)
    counter = 0
    for j in range(0, nelz):
        for i in range(0, nelx):
            rho_e_profile[j] += rho_e[counter]
            counter += 1
    rho_e_profile /= nelx

    # np.savetxt(output_folder+'/profiles/rho_n_profile.ascii',np.array([coords_n_profile,rho_n_profile]).T,header='# z,rho')
    # np.savetxt(output_folder+'/profiles/rho_e_profile.ascii',np.array([coords_e_profile,rho_e_profile]).T,header='# z,rho')

    print("     -> rho_n_profile (m,M) %.3e %.3e " % (np.min(rho_n_profile), np.max(rho_n_profile)))
    print("     -> rho_e_profile (m,M) %.3e %.3e " % (np.min(rho_e_profile), np.max(rho_e_profile)))

    print("compute rho_profile: ......................... %.3f s" % (clock.time() - start))
    timings[30] += clock.time() - start

    ###############################################################################################
    # @@ remove nodal rho profile
    ###############################################################################################
    start = clock.time()

    if remove_rho_profile:
        rho_DT_top -= rho_n_profile[nnz - 1]
        rho_DT_bot -= rho_n_profile[0]

        counter = 0
        for j in range(0, nnz):
            for i in range(0, nnx):
                rho_n[counter] -= rho_n_profile[j]
                counter += 1

        counter = 0
        for j in range(0, nelz):
            for i in range(0, nelx):
                rho_e[counter] -= rho_e_profile[j]
                counter += 1

    print("remove rho_profile: .......................... %.3f s" % (clock.time() - start))
    timings[31] += clock.time() - start

    ###############################################################################################
    # @@ assign values to quadrature points
    # rhoq, etaq, exxq, ezzq, exzq, hcondq, hcapaq, hprodq have size (nel,nq_per_element)
    ###############################################################################################
    start = clock.time()

    rhoq = np.zeros((nel, nq_per_element), dtype=np.float64)
    match particle_rho_projection:
        case "qpts":
            counterq = 0
            for iel in range(0, nel):
                for iq in range(0, nq_per_element):
                    rhoq[iel, iq] = swarm_rho[counterq]
                    counterq += 1

        case "elemental":
            for iel in range(0, nel):
                rhoq[iel, :] = rho_e[iel]
        case "nodal":
            rhoq = project_nodal_Pfield_onto_qpoints(rho_n, nq_per_element, nel, m_P, N_P, icon_V)
        case "least_squares_P1":
            for iel in range(0, nel):
                for iq in range(nq_per_element):
                    rhoq[iel, iq] = (
                        ls_rho_a[iel]
                        + ls_rho_b[iel] * (xq[iel, iq] - x_e[iel])
                        + ls_rho_c[iel] * (zq[iel, iq] - z_e[iel])
                    )
        case "least_squares_Q1":
            for iel in range(0, nel):
                for iq in range(nq_per_element):
                    rhoq[iel, iq] = (
                        ls_rho_a[iel]
                        + ls_rho_b[iel] * (xq[iel, iq] - x_e[iel])
                        + ls_rho_c[iel] * (zq[iel, iq] - z_e[iel])
                        + ls_rho_d[iel] * (xq[iel, iq] - x_e[iel]) * (zq[iel, iq] - z_e[iel])
                    )
        case _:
            exit("particle_rho_projection: unknown value")

    etaq = np.zeros((nel, nq_per_element), dtype=np.float64)
    match particle_eta_projection:
        case "qpts":
            counterq = 0
            for iel in range(0, nel):
                for iq in range(0, nq_per_element):
                    etaq[iel, iq] = swarm_eta[counterq]
                    counterq += 1
        case "elemental":
            for iel in range(0, nel):
                etaq[iel, :] = eta_e[iel]
        case "nodal":
            etaq = project_nodal_Pfield_onto_qpoints(eta_n, nq_per_element, nel, m_P, N_P, icon_V)
        case "least_squares_P1":
            for iel in range(0, nel):
                for iq in range(nq_per_element):
                    etaq[iel, iq] = (
                        ls_eta_a[iel]
                        + ls_eta_b[iel] * (xq[iel, iq] - x_e[iel])
                        + ls_eta_c[iel] * (zq[iel, iq] - z_e[iel])
                    )
        case "least_squares_Q1":
            for iel in range(0, nel):
                for iq in range(nq_per_element):
                    etaq[iel, iq] = (
                        ls_eta_a[iel]
                        + ls_eta_b[iel] * (xq[iel, iq] - x_e[iel])
                        + ls_eta_c[iel] * (zq[iel, iq] - z_e[iel])
                        + ls_eta_d[iel] * (xq[iel, iq] - x_e[iel]) * (zq[iel, iq] - z_e[iel])
                    )
        case _:
            exit("particle_eta_projection: unknown value")

    etaq_file.write("%d %.3e %.3e\n" % (istep, np.min(etaq), np.max(etaq)))
    etaq_file.flush()

    exxq = project_nodal_Vfield_onto_qpoints(exx_n, nq_per_element, nel, m_V, N_V, icon_V)
    ezzq = project_nodal_Vfield_onto_qpoints(ezz_n, nq_per_element, nel, m_V, N_V, icon_V)
    exzq = project_nodal_Vfield_onto_qpoints(exz_n, nq_per_element, nel, m_V, N_V, icon_V)
    dpdxq = project_nodal_Vfield_onto_qpoints(dpdx_n, nq_per_element, nel, m_V, N_V, icon_V)
    dpdzq = project_nodal_Vfield_onto_qpoints(dpdz_n, nq_per_element, nel, m_V, N_V, icon_V)

    if solve_T:
        Tq = project_nodal_Tfield_onto_qpoints(T, nq_per_element, nel, m_T, N_T, icon_T)
        hcapaq = project_nodal_Pfield_onto_qpoints(hcapa_n, nq_per_element, nel, m_P, N_P, icon_V)
        hcondq = project_nodal_Pfield_onto_qpoints(hcond_n, nq_per_element, nel, m_P, N_P, icon_V)
        hprodq = project_nodal_Pfield_onto_qpoints(hprod_n, nq_per_element, nel, m_P, N_P, icon_V)
        alphaq = project_nodal_Pfield_onto_qpoints(alpha_n, nq_per_element, nel, m_P, N_P, icon_V)
    else:
        Tq = np.zeros((nel, nq_per_element), dtype=np.float64)
        hcapaq = np.zeros((nel, nq_per_element), dtype=np.float64)
        hcondq = np.zeros((nel, nq_per_element), dtype=np.float64)
        hprodq = np.zeros((nel, nq_per_element), dtype=np.float64)
        alphaq = np.zeros((nel, nq_per_element), dtype=np.float64)

    if verbose_output:
       print("     -> rhoq (m,M) %.5e %.5e " % (np.min(rhoq), np.max(rhoq)))
       print("     -> etaq (m,M) %.5e %.5e " % (np.min(etaq), np.max(etaq)))

    if debug_nan and np.isnan(np.sum(etaq)):
        exit("nan found in eta_q")
    if debug_nan and np.isnan(np.sum(rhoq)):
        exit("nan found in rho_q")

    if solve_T and verbose_output:
        print("     -> Tq (m,M) %.5e %.5e " % (np.min(Tq), np.max(Tq)))
        print("     -> hcapaq (m,M) %.5e %.5e " % (np.min(hcapaq), np.max(hcapaq)))
        print("     -> hcondq (m,M) %.5e %.5e " % (np.min(hcondq), np.max(hcondq)))
        print("     -> hprodq (m,M) %.5e %.5e " % (np.min(hprodq), np.max(hprodq)))
        print("     -> alphaq (m,M) %.5e %.5e " % (np.min(alphaq), np.max(alphaq)))

    if debug_ascii:
        np.savetxt(
            "DEBUG/rhoq.ascii",
            np.array([xq.flatten(), zq.flatten(), rhoq.flatten()]).T,
            header="#x,z,rho",fmt='%1.4e'
        )
    if debug_ascii:
        np.savetxt(
            "DEBUG/etaq.ascii",
            np.array([xq.flatten(), zq.flatten(), etaq.flatten()]).T,
            header="#x,z,eta",fmt='%1.4e'
        )
    if debug_ascii and solve_T:
        np.savetxt(
            "DEBUG/Tq.ascii",
            np.array([xq.flatten(), zq.flatten(), Tq.flatten()]).T,
            header="#x,z,T",fmt='%1.4e'
        )
    if debug_ascii and solve_T:
        np.savetxt(
            "DEBUG/hcapaq.ascii",
            np.array([xq.flatten(), zq.flatten(), hcapaq.flatten()]).T,
            header="#x,z,hcapa",fmt='%1.4e'
        )
    if debug_ascii and solve_T:
        np.savetxt(
            "DEBUG/hcondq.ascii",
            np.array([xq.flatten(), zq.flatten(), hcondq.flatten()]).T,
            header="#x,z,hcond",fmt='%1.4e'
        )
    if debug_ascii and solve_T:
        np.savetxt(
            "DEBUG/hprodq.ascii",
            np.array([xq.flatten(), zq.flatten(), hprodq.flatten()]).T,
            header="#x,z,hprod",fmt='%1.4e'
        )
    if debug_ascii and solve_T:
        np.savetxt(
            "DEBUG/alphaq.ascii",
            np.array([xq.flatten(), zq.flatten(), alphaq.flatten()]).T,
            header="#x,z,alpha",fmt='%1.4e'
        )

    print("project nodal fields onto qpts: .............. %.3f s" % (clock.time() - start))
    timings[21] += clock.time() - start

    # inspect_element(26530,m_V,icon_V,x_V,z_V,rho_n,eta_n,nq_per_element,xq,zq,rhoq,etaq)

    ###############################################################################################
    # @@ compute lithostatic pressure a la Jourdon & May, Solid Earth, 2022
    # spsolve(A, b, permc_spec=None, use_umfpack=True)
    # Solve the sparse linear system Ax=b, where b may be a vector or a matrix.
    # using T structures here (bignb_T,II_T,JJ_T,VV_T) which makes sense when both V and T 
    # FE spaces are equal, but there will be a pb when they are not!
    # TODO: revisit when quad_mini is implemented!!
    ###############################################################################################
    start = clock.time()

    if compute_plith:
        VV_T, b_fem = build_matrix_plith(
            bignb_T,
            nel,
            nq_per_element,
            m_T,
            Nfem_T,
            icon_V,
            rhoq,
            gx_q,
            gz_q,
            JxWq,
            N_V,
            dNdr_V,
            dNdt_V,
            jcbi00q,
            jcbi01q,
            jcbi10q,
            jcbi11q,
            top_Vnodes,
        )
        A_fem = sparse.coo_matrix((VV_T, (II_T, JJ_T)), shape=(Nfem_T, Nfem_T)).tocsr()
        plith = sps.linalg.spsolve(A_fem, b_fem)

        print("     -> plith (m,M) %.3e %.3e " % (np.min(plith), np.max(plith)))

    else:
        plith = np.zeros(nn_V, dtype=np.float64)

    print("compute lithostatic pressure: ................ %.3f s" % (clock.time() - start))
    timings[28] += clock.time() - start

    ###############################################################################################
    # @@ build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    ###############################################################################################
    start = clock.time()

    if solve_Stokes:
        VV_Stokes, rhs_f, rhs_h, VV_K, VV_G, VV_GT, VV_M, VV_M_eta, VV_H = build_matrix_stokes(
            bignb_Stokes,
            bignb_K,
            bignb_P,
            bignb_G,
            nel,
            nq_per_element,
            m_V,
            m_P,
            ndof_V,
            Nfem_V,
            Nfem_P,
            ndof_V_el,
            icon_V,
            icon_P,
            rhoq,
            etaq,
            JxWq,
            local_to_globalV,
            gx_q,
            gz_q,
            N_V,
            N_P,
            dNdr_V,
            dNdt_V,
            dNdr_P,
            dNdt_P,
            jcbi00q,
            jcbi01q,
            jcbi10q,
            jcbi11q,
            eta_e,
            eta_ref,
            L_ref,
            bc_fix_V,
            bc_val_V,
            bot_element,
            top_element,
            bot_free_slip,
            top_free_slip,
            geometry,
            theta_V,
            axisymmetric,
            xq,
            blocks,
        )

        if debug_solver and blocks:
            print("     -> K block:", np.size(VV_K) * 8 / 1024 / 1024, "Mb")
            print("     -> G block:", np.size(VV_G) * 8 / 1024 / 1024, "Mb")
            print("     -> GT block:", np.size(VV_GT) * 8 / 1024 / 1024, "Mb")
            print("     -> MP block:", np.size(VV_M) * 8 / 1024 / 1024, "Mb")
            print("     -> MP block:", np.size(VV_M_eta) * 8 / 1024 / 1024, "Mb")
            print("     -> H block:", np.size(VV_H) * 8 / 1024 / 1024, "Mb")

    if debug_nan and np.isnan(np.sum(VV_Stokes)):
        exit("nan found in VV_Stokes")
    if debug_nan and np.isnan(np.sum(VV_M_eta)):
        exit("nan found in VV_M_eta")
    if debug_nan and np.isnan(np.sum(VV_M)):
        exit("nan found in VV_M")
    if debug_nan and np.isnan(np.sum(VV_G)):
        exit("nan found in VV_G")
    if debug_nan and np.isnan(np.sum(VV_GT)):
        exit("nan found in VV_GT")
    if debug_nan and np.isnan(np.sum(VV_K)):
        exit("nan found in VV_K")

    print("build FE matrix Stokes: ...................... %.3f s %d %d" % (clock.time() - start, Nfem, nel))
    timings[1] += clock.time() - start

    ###############################################################################################
    # @@ convert matrix arrays to coo then to csr
    # By default when converting to CSR or CSC format, duplicate (i,j) entries will be
    # summed together. This facilitates efficient construction of finite element matrices and the like.
    ###############################################################################################
    start = clock.time()

    if solve_Stokes:
        if blocks:
            K_fem = sparse.coo_matrix((VV_K, (II_K, JJ_K)), shape=(Nfem_V, Nfem_V)).tocsc()
            M_fem = sparse.coo_matrix((VV_M, (II_MP, JJ_MP)), shape=(Nfem_P, Nfem_P)).tocsc()
            M_eta_fem = sparse.coo_matrix((VV_M_eta, (II_MP, JJ_MP)), shape=(Nfem_P, Nfem_P)).tocsc()
            G_fem = sparse.coo_matrix((VV_G, (II_G, JJ_G)), shape=(Nfem_V, Nfem_P)).tocsr()
            GT_fem = sparse.coo_matrix((VV_GT, (II_GT, JJ_GT)), shape=(Nfem_P, Nfem_V)).tocsr()
            H_fem = sparse.coo_matrix((VV_H, (II_GT, JJ_GT)), shape=(Nfem_P, Nfem_V)).tocsr()
            if debug_solver:
                print("     -> block K:", np.shape(K_fem))
                print("     -> block G:", np.shape(G_fem))
                print("     -> block GT:", np.shape(GT_fem))
                print("     -> block M:", np.shape(M_fem))
                print("     -> block M_eta:", np.shape(M_eta_fem))
                print("     -> block H:", np.shape(H_fem))
        else:
            A_fem = sparse.coo_matrix((VV_Stokes, (II_Stokes, JJ_Stokes)), shape=(Nfem, Nfem)).tocsr()

    print("convert fem blocks to csr: ................... %.3f s %d %d" % (clock.time() - start, Nfem, nel))

    ###############################################################################################
    # @@ solve stokes system
    # spsolve(A, b, permc_spec=None, use_umfpack=True)
    # Solve the sparse linear system Ax=b, where b may be a vector or a matrix.
    ###############################################################################################
    start = clock.time()

    if solve_Stokes:
        if blocks:
            # sol_V,sol_P,nniter,array_xiV,array_xiP,array_alpha=\
            # uzawa3_solver_L2(K_fem,G_fem,GT_fem,M_fem,H_fem,rhs_f,rhs_h,Nfem_P)

            sol_V, sol_P, nniter, array_xiV, array_xiP, array_alpha, array_beta = cd_u_eta(
                K_fem, G_fem, GT_fem, M_fem, M_eta_fem, H_fem, rhs_f, rhs_h, Nfem_P
            )

            print("     converged in ", nniter, " iterations")
            if debug_solver:
                for k in range(0, nniter):
                    print("     iter %3d xiP= %e xiV= %e" % (k, array_xiP[k], array_xiV[k]))
            np.savetxt(
                output_folder+"/solver_convergence_" + str(istep) + ".ascii",
                np.array([array_xiV[:nniter],array_xiP[:nniter],
                        array_alpha[:nniter],array_beta[:nniter]]).T,fmt='%1.4e')
        else:
            b_fem = np.zeros(Nfem, dtype=np.float64)
            b_fem[0:Nfem_V] = rhs_f
            b_fem[Nfem_V:Nfem] = rhs_h
            sol = sps.linalg.spsolve(A_fem, b_fem)
            nniter = 0
    else:
        sol = np.zeros(Nfem, dtype=np.float64)

    print("solve Stokes system: ......................... %.3f s %d %d %d" % (clock.time() - start, Nfem, nel, nniter))
    timings[2] += clock.time() - start

    ###############################################################################################
    # @@ split solution into separate u,v,p velocity arrays
    ###############################################################################################
    start = clock.time()

    if blocks:
        u, w = np.reshape(sol_V[0:Nfem_V], (nn_V, 2)).T
        p = sol_P * (eta_ref / L_ref)
    else:
        u, w = np.reshape(sol[0:Nfem_V], (nn_V, 2)).T
        p = sol[Nfem_V:Nfem] * (eta_ref / L_ref)

    print("split sol vector into u,w,p: ................. %.3f s" % (clock.time() - start))
    timings[14] += clock.time() - start

    ###############################################################################################
    # @@ process u,w arrays
    ###############################################################################################
    start = clock.time()

    u, w, vel = process_velocity_solution_vectors(
        geometry,
        u,
        w,
        x_V,
        z_V,
        x_P,
        z_P,
        top_free_slip,
        bot_free_slip,
        top_Vnodes,
        bot_Vnodes,
        bc_fix_V,
        vel_scale,
        vel_unit,
        geo_time,
        time_scale,
        istep,
        nstep,
        nq_per_element,
        nel,
        icon_V,
        xq,
        zq,
        nn_V,
        N_V,
        JxWq,
        rad_V,
        theta_V,
        rad_P,
        theta_P,
        debug_nan,
        debug_ascii,
        every_solution_vtu,
        vstats_file,
        output_folder,
    )

    print("process u,w vectors: ................. %.3f s" % (clock.time() - start))
    timings[37] += clock.time() - start

    ###############################################################################################
    # @@ convert velocity to polar coordinates
    ###############################################################################################
    start = clock.time()

    match geometry:
        case "quarter" | "half" | "eighth" | "annulus":
            if axisymmetric:
                vr = u * np.cos(theta_V) + w * np.sin(theta_V)
                vt = u * np.sin(theta_V) - w * np.cos(theta_V)
            else:
                vr = u * np.cos(theta_V) + w * np.sin(theta_V)
                vt = -u * np.sin(theta_V) + w * np.cos(theta_V)
            if debug_ascii:
                np.savetxt(
                    "DEBUG/velocity_polar.ascii",
                    np.array([x_V, z_V, vr, vt, rad_V, theta_V]).T,
                    header="#x,z,vr,vt,rad,theta",fmt='%1.4e'
                )
                np.savetxt(
                    "DEBUG/top_vt.ascii",
                    np.array([theta_V[top_Vnodes], vt[top_Vnodes]]).T,
                    header="#theta,vt",fmt='%1.4e'
                )
                np.savetxt(
                    "DEBUG/bot_vt.ascii",
                    np.array([theta_V[bot_Vnodes], vt[bot_Vnodes]]).T,
                    header="#theta,vt",fmt='%1.4e'
                )

            print("convert velocity to polar/sph coords: %.3f s" % (clock.time() - start))
            timings[32] += clock.time() - start

        case _:
            vr = 0
            vt = 0


    ###############################################################################################
    # @@ normalise pressure: simple approach to have avrg p = 0 (volume or surface)
    # note that the surface normalisation is not super clean
    ###############################################################################################
    start = clock.time()

    if debug_nan and np.isnan(np.sum(p)):
        exit("nan found in p")

    if debug_ascii:
        np.savetxt(
            "DEBUG/pressure.ascii",
            np.array([x_P, z_P, p, rad_P, theta_P]).T,
            header="# x,z,p,rad,theta",
        )

    print("     -> p bef (m,M) %.3e %.3e %s" % (np.min(p) / p_scale, np.max(p) / p_scale, p_unit))

    pressure_offset = compute_pressure_offset(
        geometry,
        pressure_normalisation,
        axisymmetric,
        top_element,
        nel,
        nelx,
        nq_per_element,
        N_P,
        JxWq,
        p,
        icon_P,
        theta_P,
        xq,
        volume,
    )

    print("     -> pressure_offset= %.4e" % (pressure_offset))

    p -= pressure_offset

    print("     -> p aft (m,M) %.3e %.3e %s" % (np.min(p) / p_scale, np.max(p) / p_scale, p_unit))

    pstats_file.write("%d %.3e %.3e\n" % (geo_time / time_scale, np.min(p), np.max(p)))
    pstats_file.flush()

    if istep % every_solution_vtu == 0 or istep == nstep - 1:
        match geometry:
            case "box":
                np.savetxt(
                    output_folder+"/top/top_p_" + str(istep) + ".ascii",
                    np.array([x_P[top_Pnodes], p[top_Pnodes]]).T,fmt='%1.4e'
                )
                np.savetxt(
                    output_folder+"/bottom/bot_p_" + str(istep) + ".ascii",
                    np.array([x_P[bot_Pnodes], p[bot_Pnodes]]).T,fmt='%1.4e'
                )
            case "quarter" | "half" | "eighth" | "annulus":
                np.savetxt(
                    output_folder+"/top/top_p_" + str(istep) + ".ascii",
                    np.array([theta_P[top_Pnodes], p[top_Pnodes]]).T,fmt='%1.4e'
                )
                np.savetxt(
                    output_folder+"/bottom/bot_p_" + str(istep) + ".ascii",
                    np.array([theta_P[bot_Pnodes], p[bot_Pnodes]]).T,fmt='%1.4e'
                )

    if debug_ascii:
        np.savetxt(
            "DEBUG/pressure_normalised.ascii",
            np.array([x_P, z_P, p, rad_P, theta_P]).T,
            header="# x,z,p,rad,theta",fmt='%1.4e'
        )

    print("normalise pressure: .......................... %.3f s" % (clock.time() - start))
    timings[12] += clock.time() - start

    ###############################################################################################
    # @@ compute elemental pressure
    ###############################################################################################
    start = clock.time()

    p_e = np.zeros(nel, dtype=np.float64)

    for iel in range(0, nel):
        p_e[iel] = np.sum(p[icon_P[:, iel]]) / m_P

    print("     -> p_e (m,M) %.3e %.3e %s" % (np.min(p_e) / p_scale, np.max(p_e) / p_scale, p_unit))

    if istep % every_solution_vtu == 0 or istep == nstep - 1:
        match geometry:
            case "box":
                array = np.array([x_e[top_element], p_e[top_element]]).T
                np.savetxt(output_folder+"/top/top_p_e_" + str(istep) + ".ascii", array)
                array = np.array([x_e[bot_element], p_e[bot_element]]).T
                np.savetxt(output_folder+"/bottom/bot_p_e_" + str(istep) + ".ascii", array)
            case "quarter" | "half" | "eighth" | "annulus":
                array = np.array([theta_e[top_element], p_e[top_element]]).T
                np.savetxt(output_folder+"/top/top_p_e_" + str(istep) + ".ascii", array)
                array = np.array([theta_e[bot_element], p_e[bot_element]]).T
                np.savetxt(output_folder+"/bottom/bot_p_e_" + str(istep) + ".ascii", array)

    if debug_ascii:
        np.savetxt("DEBUG/pressure_e.ascii", np.array([x_e, z_e, p_e]).T, header="# x,z,p")

    print("compute elemental pressure: .................. %.3f s" % (clock.time() - start))
    timings[33] += clock.time() - start

    ###############################################################################################
    # @@ project Q1 pressure onto Q2 (vel,T) mesh
    ###############################################################################################
    start = clock.time()

    q = compute_nodal_pressure(m_V, nn_V, icon_V, icon_P, p, N_P_n)

    print("     -> q (m,M) %.3e %.3e %s" % (np.min(q), np.max(q), p_unit))

    if debug_ascii:
        np.savetxt("DEBUG/q.ascii", np.array([x_V, z_V, q]).T, header="# x,z,q")

    if istep % every_solution_vtu == 0 or istep == nstep - 1:
        match geometry:
            case "box":
                np.savetxt(
                    output_folder+"/top/top_q_" + str(istep) + ".ascii",
                    np.array([x_V[top_Vnodes], q[top_Vnodes]]).T,
                )
                np.savetxt(
                    output_folder+"/bottom/bot_q_" + str(istep) + ".ascii",
                    np.array([x_V[bot_Vnodes], q[bot_Vnodes]]).T,
                )
            case "quarter" | "half" | "eighth" | "annulus":
                np.savetxt(
                    output_folder+"/top/top_q_" + str(istep) + ".ascii",
                    np.array([theta_V[top_Vnodes], q[top_Vnodes]]).T,
                )
                np.savetxt(
                    output_folder+"/bottom/bot_q_" + str(istep) + ".ascii",
                    np.array([theta_V[bot_Vnodes], q[bot_Vnodes]]).T,
                )

    print("compute nodal press: ......................... %.3f s" % (clock.time() - start))
    timings[3] += clock.time() - start

    ###########################################################################

    if nonlinear and iter_nl<niter_nl-1:
          inside_nonlinear_iterations=True
    else:
       inside_nonlinear_iterations=False
       if iter_nl==niter_nl-1:
          print(" maximum number of nonlinear iterations reached")

    ###############################################################################################
    # @@ assess convergence of nonlinear iterations
    ###############################################################################################
    start = clock.time()

    if nonlinear:
       u_mem, w_mem, p_mem, T_mem, inside_nonlinear_iterations = \
       assess_nlconvergence(istep, iter_nl, solve_Stokes, solve_T, \
       u, w, p, T, u_mem, w_mem, p_mem, T_mem, tol_nl, inside_nonlinear_iterations, conv_file)
    else:
        T_mem = T.copy()

    print("assess steady state: ........................ %.4f s" % (clock.time() - start))
    timings[38] += clock.time() - start

    ###############################################################################################
    # @@ compute timestep
    # note that the timestep is not allowed to increase by more than 25% in one go
    ###############################################################################################
    start = clock.time()

    if solve_Stokes:
        match geometry:
            case "box":
                dt1 = CFLnb * min(hx, hz) / np.max(vel)
            case "quarter" | "half" | "eighth" | "annulus":
                dt1 = CFLnb * hrad / np.max(vel)
    else:
        dt1 = 0.0

    print("     -> dt1= %.3e %s" % (dt1 / time_scale, time_unit))

    if solve_T:
        avrg_hcond = np.average(swarm_hcond)
        avrg_hcapa = np.average(swarm_hcapa)
        avrg_rho = np.average(swarm_rho)
        match geometry:
            case "box":
                dt2 = CFLnb * min(hx, hz) ** 2 / (avrg_hcond / avrg_hcapa / avrg_rho)
            case "quarter" | "half" | "eighth" | "annulus":
                dt2 = CFLnb * hrad**2 / (avrg_hcond / avrg_hcapa / avrg_rho)
        print("     -> dt2= %.3e %s" % (dt2 / time_scale, time_unit))
    else:
        dt2 = 1e50

    dt1 = min(dt1, 1.25 * dt1_mem)  # limiter
    dt2 = min(dt2, 1.25 * dt2_mem)  # limiter

    dt = np.min([dt1, dt2, dt_max])

    if not inside_nonlinear_iterations:

       #geo_time += dt #moved at the end

       print("     -> dt = %.3e %s" % (dt / time_scale, time_unit))
       print("     -> geological time = %e %s" % (geo_time / time_scale, time_unit))

       dt_file.write("%e %e %e %e\n" % (geo_time / time_scale, dt / time_scale, dt1 / time_scale, dt2 / time_scale))
       dt_file.flush()

       dt1_mem = dt1
       dt2_mem = dt2

    print("compute time step: ........................... %.3f s" % (clock.time() - start))
    timings[19] += clock.time() - start

    ###############################################################################################
    # @@ project velocity on quadrature points
    ###############################################################################################
    start = clock.time()

    uq = project_nodal_Vfield_onto_qpoints(u, nq_per_element, nel, m_V, N_V, icon_V)
    wq = project_nodal_Vfield_onto_qpoints(w, nq_per_element, nel, m_V, N_V, icon_V)
    pq = project_nodal_Pfield_onto_qpoints(p, nq_per_element, nel, m_P, N_P, icon_P)

    print("project u,v,p on quad points: ................ %.3f s" % (clock.time() - start))
    timings[21] += clock.time() - start

    ###############################################################################################
    # @@ compute L2 errors
    ###############################################################################################
    start = clock.time()

    if compute_L2_errors:
        errv, errp = compute_discretisation_errors(nel, nq_per_element, xq, zq, uq, wq, pq, volume, JxWq, experiment)

        print("     -> errv,errp= %.3e %.3e %d" % (errv, errp, nel))

    print("compute discretisation errors: ............... %.3f s" % (clock.time() - start))

    ###############################################################################################
    # @@ build temperature matrix
    ###############################################################################################
    start = clock.time()

    if solve_T:
        VV_T, rhs = build_matrix_energy(
            bignb_T,
            nel,
            nq_per_element,
            m_T,
            Nfem_T,
            T_mem,   # NEW 23june
            icon_T,
            rhoq,
            etaq,
            Tq,
            uq,
            wq,
            hcondq,
            hcapaq,
            alphaq,
            hprodq,
            exxq,
            ezzq,
            exzq,
            dpdxq,
            dpdzq,
            JxWq,
            N_T,
            dNdr_T,
            dNdt_T,
            jcbi00q,
            jcbi01q,
            jcbi10q,
            jcbi11q,
            bc_fix_T,
            bc_val_T,
            dt,
            formulation,
            rho0,
        )

        print("build FE matrix : ............................ %.3f s" % (clock.time() - start))
        timings[4] += clock.time() - start

    ###############################################################################################
    # @@ solve temperature system
    ###############################################################################################
    start = clock.time()

    if solve_T:
        sparse_matrix = sparse.coo_matrix((VV_T, (II_T, JJ_T)), shape=(Nfem_T, Nfem_T)).tocsr()

        T = sps.linalg.spsolve(sparse_matrix, rhs)

        if debug_nan and np.isnan(np.sum(T)):
            exit("nan found in T")

        print("     -> T (m,M) %.3e %.3e " % (np.min(T), np.max(T)))

        if debug_ascii:
            np.savetxt("DEBUG/T.ascii", np.array([x_T, z_T, T]).T, header="# x,z,T")

        Tstats_file.write("%.3e %.3e %.3e\n" % (geo_time / time_scale, np.min(T) - TKelvin, np.max(T) - TKelvin))
        Tstats_file.flush()

        print("solve T time: ................................ %.3f s" % (clock.time() - start))
        timings[5] += clock.time() - start

    # end if solve_T

    ###############################################################################################
    # @@ compute nodal heat flux
    # ordering 0-1-2-3 is BL-BR-TR-TL
    ###############################################################################################
    start = clock.time()

    if solve_T:
        dTdx_n, dTdz_n, qx_n, qz_n = compute_nodal_heat_flux(
            icon_T,T,hcond_n,nn_T,m_T,nel,dNdr_T_n,dNdt_T_n,
            jcbi00_T,jcbi01_T,jcbi10_T,jcbi11_T)

        print("     -> dTdx_n (m,M) %.3e %.3e " % (np.min(dTdx_n), np.max(dTdx_n)))
        print("     -> dTdz_n (m,M) %.3e %.3e " % (np.min(dTdz_n), np.max(dTdz_n)))
        print("     -> qx_n (m,M) %.3e %.3e " % (np.min(qx_n), np.max(qx_n)))
        print("     -> qz_n (m,M) %.3e %.3e " % (np.min(qz_n), np.max(qz_n)))

        qx0 = qx_n[cornerBL]
        qz0 = qz_n[cornerBL]
        qx1 = qx_n[cornerBR]
        qz1 = qz_n[cornerBR]
        qx2 = qx_n[cornerTR]
        qz2 = qz_n[cornerTR]
        qx3 = qx_n[cornerTL]
        qz3 = qz_n[cornerTL]

        corner_q_file.write(
            "%e %e %e %e %e %e %e %e %e\n" % (geo_time / time_scale, qx0, qz0, qx1, qz1, qx2, qz2, qx3, qz3)
        )
        corner_q_file.flush()

        print("compute nodal heat flux: ..................... %.3f s" % (clock.time() - start))
        timings[7] += clock.time() - start

    else:
        qx_n = 0
        qz_n = 0

    ###############################################################################################
    # @@ compute heat flux and Nusselt at top and bottom
    ###############################################################################################
    start = clock.time()

    if solve_T:
        avrg_T_bot, avrg_T_top, avrg_dTdz_bot, avrg_dTdz_top, Nu = compute_Nu(
            Lx,
            Lz,
            nel,
            top_element,
            bot_element,
            icon_V,
            T,
            dTdz_n,
            nq_per_dim,
            qcoords,
            qweights,
            hx,
        )

        print("     -> <T> (bot,top)= %.3e %.3e " % (avrg_T_bot, avrg_T_top))
        print("     -> <dTdz> (bot,top)= %.3e %.3e " % (avrg_dTdz_bot, avrg_dTdz_top))
        print("     -> Nusselt= %.3e " % (Nu))

        Nu_file.write("%e %.6e \n" % (geo_time / time_scale, Nu))
        Nu_file.flush()
        avrg_T_bot_file.write("%e %e \n" % (geo_time / time_scale, avrg_T_bot))
        avrg_T_bot_file.flush()
        avrg_T_top_file.write("%e %e \n" % (geo_time / time_scale, avrg_T_top))
        avrg_T_top_file.flush()
        avrg_dTdz_bot_file.write("%e %e \n" % (geo_time / time_scale, avrg_dTdz_bot))
        avrg_dTdz_bot_file.flush()
        avrg_dTdz_top_file.write("%e %e \n" % (geo_time / time_scale, avrg_dTdz_top))
        avrg_dTdz_top_file.flush()

        print("compute q and Nu at top & bottom: ............ %.3f s" % (clock.time() - start))
        timings[8] += clock.time() - start

    ###############################################################################################
    # compute elemental strain rate and deviatoric strainrate
    ###############################################################################################
    start = clock.time()

    exx_e, ezz_e, exz_e = compute_elemental_strain_rate(icon_V, u, w, nn_V, nel, x_V, z_V)

    divv_e = exx_e + ezz_e
    dxx_e = exx_e - divv_e / 3
    dzz_e = ezz_e - divv_e / 3
    dxz_e = exz_e

    if verbose_output:
       print("     -> exx_e (m,M) %.3e %.3e " % (np.min(exx_e), np.max(exx_e)))
       print("     -> ezz_e (m,M) %.3e %.3e " % (np.min(ezz_e), np.max(ezz_e)))
       print("     -> exz_e (m,M) %.3e %.3e " % (np.min(exz_e), np.max(exz_e)))
       print("     -> dxx_e (m,M) %.3e %.3e " % (np.min(dxx_e), np.max(dxx_e)))
       print("     -> dzz_e (m,M) %.3e %.3e " % (np.min(dzz_e), np.max(dzz_e)))
       print("     -> dxz_e (m,M) %.3e %.3e " % (np.min(dxz_e), np.max(dxz_e)))

    if debug_ascii:
        np.savetxt(
            "DEBUG/strainrate_cartesian_e.ascii",
            np.array([x_e, z_e, exx_e, ezz_e, exz_e, effective(exx_e, ezz_e, exz_e)]).T,
            header="#x,z,exx,ezz,exz,e",
        )

    match geometry:
        case "quarter" | "half" | "eighth" | "annulus":
            if axisymmetric:
                err_e, ett_e, ert_e = convert_tensor_to_spherical_coords(theta_e, exx_e, ezz_e, exz_e)
                drr_e, dtt_e, drt_e = convert_tensor_to_spherical_coords(theta_e, dxx_e, dzz_e, dxz_e)
            else:
                err_e, ett_e, ert_e = convert_tensor_to_polar_coords(theta_e, exx_e, ezz_e, exz_e)
                drr_e, dtt_e, drt_e = convert_tensor_to_polar_coords(theta_e, dxx_e, dzz_e, dxz_e)

            print("     -> err_e (m,M) %.3e %.3e " % (np.min(err_e), np.max(err_e)))
            print("     -> ett_e (m,M) %.3e %.3e " % (np.min(ett_e), np.max(ett_e)))
            print("     -> ert_e (m,M) %.3e %.3e " % (np.min(ert_e), np.max(ert_e)))

            if debug_ascii:
                np.savetxt(
                    "DEBUG/strainrate_polar_e.ascii",
                    np.array([x_e, z_e, err_e, ett_e, ert_e]).T,
                    header="#x,z,err,ett,ert",
                )

            if istep % every_solution_vtu == 0 or istep == nstep - 1:
                np.savetxt(
                    output_folder+"/top/top_err_e_" + str(istep) + ".ascii",
                    np.array([theta_e[top_element], err_e[top_element]]).T,
                )
                np.savetxt(
                    output_folder+"/top/top_drr_e_" + str(istep) + ".ascii",
                    np.array([theta_e[top_element], drr_e[top_element]]).T,
                )

    print("compute elemental sr: ........................ %.3f s" % (clock.time() - start))
    timings[29] += clock.time() - start

    ###############################################################################################
    # @@ compute nodal strainrate
    # method 1 default, method 2 is probably bit more accurate, but more expensive
    ###############################################################################################
    start = clock.time()

    if method_nodal_strain_rate == 1:
        exx_n, ezz_n, exz_n = compute_nodal_strain_rate(
            icon_V, u, w, nn_V, m_V, nel, dNdr_V_n, dNdt_V_n, jcbi00n, jcbi01n, jcbi10n, jcbi11n
        )

    if method_nodal_strain_rate == 2:
        exx_n, ezz_n, exz_n = compute_nodal_strain_rate2(
            bignb_T,
            II_T,
            JJ_T,
            m_T,
            nq_per_element,
            icon_V,
            u,
            w,
            nn_V,
            nel,
            JxWq,
            N_V,
            dNdr_V,
            dNdt_V,
            jcbi00q,
            jcbi01q,
            jcbi10q,
            jcbi11q,
        )

    e_n = effective(exx_n, ezz_n, exz_n)

    if verbose_output:
       print("     -> exx_n (m,M) %.3e %.3e " % (np.min(exx_n), np.max(exx_n)))
       print("     -> ezz_n (m,M) %.3e %.3e " % (np.min(ezz_n), np.max(ezz_n)))
       print("     -> exz_n (m,M) %.3e %.3e " % (np.min(exz_n), np.max(exz_n)))

    srstats_file.write("%e %e %e\n" % (geo_time / time_scale, np.min(e_n), np.max(e_n)))
    srstats_file.flush()

    if debug_ascii:
        np.savetxt(
            "DEBUG/strainrate_cartesian_n.ascii",
            np.array([x_V, z_V, exx_n, ezz_n, exz_n, e_n, rad_V, theta_V]).T,
            header="#x,z,exx,ezz,exz,e,rad,theta",
        )

    divv_n = exx_n + ezz_n
    dxx_n = exx_n - divv_n / 3
    dzz_n = ezz_n - divv_n / 3
    dxz_n = exz_n

    if verbose_output:
       print("     -> divv_n (m,M) %.3e %.3e " % (np.min(divv_n), np.max(divv_n)))
       print("     -> dxx_n (m,M) %.3e %.3e " % (np.min(dxx_n), np.max(dxx_n)))
       print("     -> dzz_n (m,M) %.3e %.3e " % (np.min(dzz_n), np.max(dzz_n)))
       print("     -> dxz_n (m,M) %.3e %.3e " % (np.min(dxz_n), np.max(dxz_n)))

    match geometry:
        case "box":
            err_n = 0
            ett_n = 0
            ert_n = 0
            if istep % every_solution_vtu == 0 or istep == nstep - 1:
                np.savetxt(
                    output_folder+"/top/top_ezz_n" + str(istep) + ".ascii",
                    np.array([x_V[top_Vnodes], ezz_n[top_Vnodes]]).T,
                )
                np.savetxt(
                    output_folder+"/bottom/bot_ezz_n" + str(istep) + ".ascii",
                    np.array([x_V[bot_Vnodes], ezz_n[bot_Vnodes]]).T,
                )

        case "quarter" | "half" | "eighth" | "annulus":
            if axisymmetric:
                err_n, ett_n, ert_n = convert_tensor_to_spherical_coords(theta_V, exx_n, ezz_n, exz_n)
                if debug_ascii:
                    np.savetxt(
                        "DEBUG/strainrate_spherical_coords.ascii",
                        np.array([x_V, z_V, err_n, ett_n, ert_n, rad_V, theta_V]).T,
                        header="#x,z,err,ett,ert,rad,theta",
                    )
            else:
                err_n, ett_n, ert_n = convert_tensor_to_polar_coords(theta_V, exx_n, ezz_n, exz_n)
                if debug_ascii:
                    np.savetxt(
                        "DEBUG/strainrate_polar_coords.ascii",
                        np.array([x_V, z_V, err_n, ett_n, ert_n, rad_V, theta_V]).T,
                        header="#x,z,err,ett,ert,rad,theta",
                    )

            if verbose_output:
               print("     -> err_n (m,M) %.3e %.3e " % (np.min(err_n), np.max(err_n)))
               print("     -> ett_n (m,M) %.3e %.3e " % (np.min(ett_n), np.max(ett_n)))
               print("     -> ert_n (m,M) %.3e %.3e " % (np.min(ert_n), np.max(ert_n)))

            if istep % every_solution_vtu == 0 or istep == nstep - 1:
                np.savetxt(
                    output_folder+"/top/top_err_n_" + str(istep) + ".ascii",
                    np.array([theta_V[top_Vnodes], err_n[top_Vnodes]]).T,
                )
                np.savetxt(
                    output_folder+"/bottom/bot_err_n_" + str(istep) + ".ascii",
                    np.array([theta_V[bot_Vnodes], err_n[bot_Vnodes]]).T,
                )

    print("compute nodal strainrate: .................... %.3f s" % (clock.time() - start))
    timings[11] += clock.time() - start

    ###############################################################################################
    # @@ compute sr and dev sr on qpts
    ###############################################################################################
    start = clock.time()

    exxq = project_nodal_Vfield_onto_qpoints(exx_n, nq_per_element, nel, m_V, N_V, icon_V)
    ezzq = project_nodal_Vfield_onto_qpoints(ezz_n, nq_per_element, nel, m_V, N_V, icon_V)
    exzq = project_nodal_Vfield_onto_qpoints(exz_n, nq_per_element, nel, m_V, N_V, icon_V)

    dxxq = project_nodal_Vfield_onto_qpoints(dxx_n, nq_per_element, nel, m_V, N_V, icon_V)
    dzzq = project_nodal_Vfield_onto_qpoints(dzz_n, nq_per_element, nel, m_V, N_V, icon_V)
    dxzq = project_nodal_Vfield_onto_qpoints(dxz_n, nq_per_element, nel, m_V, N_V, icon_V)

    print("compute strainrate on qpts: .................. %.3f s" % (clock.time() - start))  
    #timings[11]+=clock.time()-start

    ###############################################################################################
    # @@ compute global quantities
    ###############################################################################################
    start = clock.time()

    vrms, EK, WAG, TVD, GPE, ITE, TM, T_avrg, eta_avrg = compute_global_quantities(
        nel,
        nq_per_element,
        xq,
        zq,
        uq,
        wq,
        Tq,
        rhoq,
        hcapaq,
        alphaq,
        etaq,
        exxq,
        ezzq,
        exzq,
        dxxq,
        dzzq,
        dxzq,
        volume,
        JxWq,
        gx_q,
        gz_q,
    )

    delta = WAG + TVD  # see tosn15

    vrms_file.write("%.4e %.6e \n" % (geo_time / time_scale, vrms / vel_scale))
    vrms_file.flush()
    TM_file.write("%.4e %.4e \n" % (geo_time / time_scale, TM))
    TM_file.flush()
    EK_file.write("%.4e %.4e \n" % (geo_time / time_scale, EK))
    EK_file.flush()
    TVD_file.write("%.4e %.4e \n" % (geo_time / time_scale, TVD))
    TVD_file.flush()
    WAG_file.write("%.4e %.4e \n" % (geo_time / time_scale, WAG))
    WAG_file.flush()
    delta_file.write("%.4e %.4e %.4e\n" % (geo_time / time_scale, delta, max(abs(WAG), TVD)))
    delta_file.flush()
    T_avrg_file.write("%.4e %.6e \n" % (geo_time / time_scale, T_avrg))
    T_avrg_file.flush()
    eta_avrg_file.write("%.4e %.4e \n" % (geo_time / time_scale, eta_avrg))
    eta_avrg_file.flush()

    print("     istep= %.6d ; vrms   = %.3e %s" % (istep, vrms / vel_scale, vel_unit))

    print("compute global quantities: ................... %.3f s" % (clock.time() - start))
    timings[6] += clock.time() - start

    ###############################################################################################
    # @@ compute boundary velocity statistics
    ###############################################################################################
    start = clock.time()

    vel_max_left,vel_max_right,vel_max_bottom,vel_max_top=\
    compute_boundary_velocity_statistics(x_V,z_V,u,w,left_Vnodes,right_Vnodes,bot_Vnodes,top_Vnodes)

    bc_vel_file.write("%.4e %.4e %.4e %.4e %.4e  \n" % (geo_time / time_scale,\
                                                        vel_max_left / vel_scale,\
                                                        vel_max_right / vel_scale,\
                                                        vel_max_bottom / vel_scale,\
                                                        vel_max_top / vel_scale)) 

    print("compute boundary vel stats: .................. %.3f s" % (clock.time() - start))
    #timings[6] += clock.time() - start


    ###############################################################################################
    # @@ compute deviatoric stress tensor components (elemental & nodal)
    ###############################################################################################
    start = clock.time()

    (
        tauxx_n,
        tauzz_n,
        tauxz_n,
        tauxx_e,
        tauzz_e,
        tauxz_e,
        taurr_n,
        tautt_n,
        taurt_n,
        taurr_e,
        tautt_e,
        taurt_e,
    ) = compute_deviatoric_stress_tensor(
        solve_Stokes,
        geometry,
        x_V,
        theta_V,
        eta_n,
        x_e,
        theta_e,
        eta_e,
        istep,
        nstep,
        every_solution_vtu,
        axisymmetric,
        dxx_n,
        dzz_n,
        dxz_n,
        dxx_e,
        dzz_e,
        dxz_e,
        top_Vnodes,
        bot_Vnodes,
        top_element,
        bot_element,
        verbose_output,
        output_folder,
    )

    print("compute deviatoric stress: ................... %.3f s" % (clock.time() - start))
    timings[27] += clock.time() - start

    ###############################################################################################
    # @@ compute full stress tensor components
    ###############################################################################################
    start = clock.time()

    (
        sigmaxx_n,
        sigmazz_n,
        sigmaxz_n,
        sigmaxx_e,
        sigmazz_e,
        sigmaxz_e,
        sigmarr_n,
        sigmatt_n,
        sigmart_n,
        sigmarr_e,
        sigmatt_e,
        sigmart_e,
    ) = compute_full_stress_tensor(
        solve_Stokes,
        geometry,
        x_V,
        theta_V,
        q,
        x_e,
        theta_e,
        p_e,
        istep,
        nstep,
        every_solution_vtu,
        axisymmetric,
        tauxx_n,
        tauzz_n,
        tauxz_n,
        tauxx_e,
        tauzz_e,
        tauxz_e,
        top_Vnodes,
        bot_Vnodes,
        top_element,
        bot_element,
        output_folder,
    )

    print("compute full stress: ......................... %.3f s" % (clock.time() - start))
    timings[27] += clock.time() - start

    ###############################################################################################
    # @@ compute dynamic topography at bottom and surface topo
    ###############################################################################################
    start = clock.time()

    if solve_Stokes and compute_dynamic_topography:
        match geometry:
            case "box":
                #
                avrg_sigmazz = np.average(sigmazz_n[top_Vnodes])
                dyn_topo_top = (
                    (sigmazz_n[top_Vnodes] - avrg_sigmazz) / gz_n[top_Vnodes] / (rho_n[top_Vnodes] - rho_DT_top)
                )
                np.savetxt(
                    output_folder+"/top/top_dynamic_topography_n_" + str(istep) + ".ascii",
                    np.array([x_V[top_Vnodes], dyn_topo_top]).T,
                )
                #
                avrg_sigmazz = np.average(sigmazz_n[bot_Vnodes])
                dyn_topo_bot = (
                    (sigmazz_n[bot_Vnodes] - avrg_sigmazz) / gz_n[bot_Vnodes] / (rho_n[bot_Vnodes] - rho_DT_bot)
                )
                np.savetxt(
                    output_folder+"/bottom/bot_dynamic_topography_n_" + str(istep) + ".ascii",
                    np.array([x_V[bot_Vnodes], dyn_topo_bot]).T,
                )

            case "quarter" | "half" | "eightgh":
                #
                avrg_sigmarr = np.average(sigmarr_n[top_Vnodes])
                dyn_topo_top = (
                    (sigmarr_n[top_Vnodes] - avrg_sigmarr) / gr_n[top_Vnodes] / (rho_n[top_Vnodes] - rho_DT_top)
                )
                np.savetxt(
                    output_folder+"/top/top_dynamic_topography_n_" + str(istep) + ".ascii",
                    np.array([theta_V[top_Vnodes], dyn_topo_top]).T,
                )
                #
                avrg_sigmarr = np.average(sigmarr_e[top_element])
                dyn_topo_top = (
                    (sigmarr_e[top_element] - avrg_sigmarr) / gr_e[top_element] / (rho_e[top_element] - rho_DT_top)
                )
                np.savetxt(
                    output_folder+"/top/top_dynamic_topography_e_" + str(istep) + ".ascii",
                    np.array([theta_e[top_element], dyn_topo_top]).T,
                )
                #
                avrg_sigmarr = np.average(sigmarr_n[bot_Vnodes])
                dyn_topo_bot = (
                    (sigmarr_n[bot_Vnodes] - avrg_sigmarr) / gr_n[bot_Vnodes] / (rho_n[bot_Vnodes] - rho_DT_bot)
                )
                np.savetxt(
                    output_folder+"/bottom/bot_dynamic_topography_n_" + str(istep) + ".ascii",
                    np.array([theta_V[bot_Vnodes], dyn_topo_bot]).T,
                )
                #
                avrg_sigmarr = np.average(sigmarr_e[bot_element])
                dyn_topo_bot = (
                    (sigmarr_e[bot_element] - avrg_sigmarr) / gr_e[bot_element] / (rho_e[bot_element] - rho_DT_bot)
                )
                np.savetxt(
                    output_folder+"/bottom/bot_dynamic_topography_e_" + str(istep) + ".ascii",
                    np.array([theta_e[bot_element], dyn_topo_bot]).T,
                )

            case _:
                raise ValueError("compute_dynamic_topography: unknown geometry")

    print("compute dynamic topo: ........................ %.3f s" % (clock.time() - start))
    timings[26] += clock.time() - start

    ###############################################################################################
    # @@ compute nodal pressure gradient
    ###############################################################################################
    start = clock.time()

    if solve_Stokes:
        dpdx_n, dpdz_n = compute_nodal_pressure_gradient(
            icon_V, q, nn_V, m_V, nel, dNdr_V_n, dNdt_V_n, jcbi00n, jcbi01n, jcbi10n, jcbi11n
        )

    print("     -> dpdx_n (m,M) %.3e %.3e " % (np.min(dpdx_n), np.max(dpdx_n)))
    print("     -> dpdz_n (m,M) %.3e %.3e " % (np.min(dpdz_n), np.max(dpdz_n)))

    if debug_ascii:
        array = np.array([x_V, z_V, dpdx_n, dpdz_n]).T
        np.savetxt("DEBUG/pressure_gradient.ascii", array, header="#x,z,dpdx,dpdz")

    print("compute nodal pressure gradient: ............. %.3f s" % (clock.time() - start))
    timings[8] += clock.time() - start

    ###############################################################################################
    # @@ advect particles
    ###############################################################################################
    start = clock.time()

    if solve_Stokes and not inside_nonlinear_iterations:
        match geometry:
            case "box":
                if use_free_surface:
                   swarm_x,swarm_z,swarm_u,swarm_w,swarm_active = advect_particles___box_fs(
                   RKorder,dt,nparticle,swarm_x,swarm_z,swarm_active,u,w,Lx,Lz,hx,hz,nelx,icon_V,x_V,z_V)
                else:
                   swarm_x, swarm_z, swarm_u, swarm_w, swarm_active = advect_particles___box(
                   RKorder,dt,nparticle,swarm_x,swarm_z,swarm_active,u,w,Lx,Lz,hx,hz,nelx,icon_V,x_V,z_V)

            case "quarter":
                swarm_x, swarm_z, swarm_rad, swarm_theta, swarm_u, swarm_w, swarm_active = advect_particles___quarter(
                    RKorder,
                    dt,
                    nparticle,
                    swarm_x,
                    swarm_z,
                    swarm_rad,
                    swarm_theta,
                    swarm_active,
                    u,
                    w,
                    Rinner,
                    Router,
                    hrad,
                    htheta,
                    nelx,
                    icon_V,
                    rad_V,
                    theta_V,
                )
            case "half":
                swarm_x, swarm_z, swarm_rad, swarm_theta, swarm_u, swarm_w, swarm_active = advect_particles___half(
                    RKorder,
                    dt,
                    nparticle,
                    swarm_x,
                    swarm_z,
                    swarm_rad,
                    swarm_theta,
                    swarm_active,
                    u,
                    w,
                    Rinner,
                    Router,
                    hrad,
                    htheta,
                    nelx,
                    icon_V,
                    rad_V,
                    theta_V,
                )
            case "eighth":
                swarm_x, swarm_z, swarm_rad, swarm_theta, swarm_u, swarm_w, swarm_active = advect_particles___eighth(
                    RKorder,
                    dt,
                    nparticle,
                    swarm_x,
                    swarm_z,
                    swarm_rad,
                    swarm_theta,
                    swarm_active,
                    u,
                    w,
                    Rinner,
                    Router,
                    hrad,
                    htheta,
                    nelx,
                    icon_V,
                    rad_V,
                    theta_V,
                )
            case "annulus":
                swarm_x, swarm_z, swarm_rad, swarm_theta, swarm_u, swarm_w, swarm_active = advect_particles___annulus(
                    RKorder,
                    dt,
                    nparticle,
                    swarm_x,
                    swarm_z,
                    swarm_rad,
                    swarm_theta,
                    swarm_active,
                    u,
                    w,
                    Rinner,
                    Router,
                    hrad,
                    htheta,
                    nelx,
                    icon_V,
                    rad_V,
                    theta_V,
                )

        if debug_ascii:
            np.savetxt("DEBUG/swarm.ascii", np.array([swarm_x, swarm_z]).T, header="#x,z")

        if verbose_output:
           print("     -> nb inactive particles:", nparticle - np.sum(swarm_active))
           print("     -> swarm_x (m,M) %.3e %.3e " % (np.min(swarm_x), np.max(swarm_x)))
           print("     -> swarm_z (m,M) %.3e %.3e " % (np.min(swarm_z), np.max(swarm_z)))
           print("     -> swarm_u (m,M) %.3e %.3e " % (np.min(swarm_u), np.max(swarm_u)))
           print("     -> swarm_w (m,M) %.3e %.3e " % (np.min(swarm_w), np.max(swarm_w)))

    else:
        swarm_u = np.zeros(nparticle, dtype=np.float64)
        swarm_w = np.zeros(nparticle, dtype=np.float64)

    print("advect particles: ............................ %.3f s" % (clock.time() - start))
    timings[13] += clock.time() - start

    ###############################################################################################
    ###############################################################################################
    start = clock.time()

    if use_free_surface and not inside_nonlinear_iterations:
        print('calling evolve mesh')
        evolve_mesh(nelx,nelz,u,w,x_V,z_V,z_P,z_T,icon_V,icon_P,top_Vnodes,m_V,N_V,dNdr_V,dNdt_V,nq_per_dim,weightq)

    print("evolve mesh: ............................ %.3f s" % (clock.time() - start))
    #timings[13] += clock.time() - start

    ###############################################################################################
    # @@ population control
    ###############################################################################################
    start = clock.time()

    if RKorder>0 and not inside_nonlinear_iterations and allow_population_control:
       population_control(istep, x_V, z_V, icon_V, nel, nparticle_min, nparticle, nparticle_e, swarm_active,\
                          swarm_id, swarm_iel, swarm_r, swarm_t, swarm_x, swarm_z, swarm_u, swarm_w, \
                          swarm_strain, swarm_eta, swarm_wf, swarm_F, \
                          swarm_paint, use_melting, nmat, ptcl_active_file,output_folder)

    print("population control: .......................... %.3f s" % (clock.time() - start))
    timings[39] += clock.time() - start

    ###############################################################################################
    # @@ locate particles and compute reduced coordinates
    ###############################################################################################
    start = clock.time()

    if RKorder>0:
       swarm_r, swarm_t, swarm_iel = locate_particles(geometry,nparticle,swarm_active,swarm_x,swarm_z,
                                                      swarm_rad,swarm_theta,hx,hz,hrad,htheta,
                                                      x_V,z_V,rad_V,theta_V,icon_V,nelx,Rinner)

    print("locate particles: ............................ %.3f s" % (clock.time() - start))
    timings[16] += clock.time() - start

    ###############################################################################################
    # @@ compute strain on particles
    ###############################################################################################
    start = clock.time()

    if not inside_nonlinear_iterations:
       swarm_strain += np.sqrt(0.5 * (swarm_exx**2 + swarm_ezz**2) + swarm_exz**2) * dt

    print("     -> swarm_strain (m,M) %.3e %.3e " % (np.min(swarm_strain), np.max(swarm_strain)))

    print("update strain on particles: .................. %.3f s" % (clock.time() - start))

    ###############################################################################################
    # @@ output min/max coordinates of each material in one single file
    # THIS IS BROKEN bc swarm_mats -> swarm_weight_fractions
    ###############################################################################################
    # start=clock.time()
    # imat=np.min(swarm_mat)
    # jmat=np.max(swarm_mat)
    # mats=np.zeros(4*(jmat-imat+1)+1,dtype=np.float64)
    # mats[0]=geo_time/time_scale
    # counter=1
    # for i in range(imat,jmat+1):
    #    xmin=np.min(swarm_x[swarm_mat==i]) ; mats[counter]=xmin ; counter+=1
    #    xmax=np.max(swarm_x[swarm_mat==i]) ; mats[counter]=xmax ; counter+=1
    #    zmin=np.min(swarm_z[swarm_mat==i]) ; mats[counter]=zmin ; counter+=1
    #    zmax=np.max(swarm_z[swarm_mat==i]) ; mats[counter]=zmax ; counter+=1
    # mats.tofile(mats_file,sep=' ',format='%.4e ') ; mats_file.write('\n')
    # mats_file.flush()
    # print("write min/max extents: ....................... %.3f s" % (clock.time()-start))
    # timings[16]+=clock.time()-start

    ###############################################################################################
    # @@ generate/write in pvd files
    ###############################################################################################

    if not inside_nonlinear_iterations:
       write_in_pvd_files(pvd_solution_file, pvd_swarm_file, istep, nstep, every_solution_vtu, every_swarm_vtu, geo_time)

    ###############################################################################################
    # @@ output solution to vtu file
    ###############################################################################################
    start = clock.time()

    if istep % every_solution_vtu == 0 or istep == nstep - 1:
        output_solution_to_vtu(
            solve_Stokes,
            istep,
            nel,
            nn_V,
            m_V,
            solve_T,
            vel_scale,
            vel_unit,
            TKelvin,
            x_V,
            z_V,
            u,
            w,
            q,
            T,
            eta_n,
            rho_n,
            exx_n,
            ezz_n,
            exz_n,
            e_n,
            divv_n,
            qx_n,
            qz_n,
            rho_e,
            exx_e,
            ezz_e,
            exz_e,
            divv_e,
            tauxx_n,
            tauzz_n,
            tauxz_n,
            sigmaxx_n,
            sigmazz_n,
            sigmaxz_n,
            rad_V,
            theta_V,
            eta_e,
            nparticle_e,
            area,
            icon_V,
            bc_fix_V,
            bc_fix_T,
            geometry,
            gx_n,
            gz_n,
            err_n,
            ett_n,
            ert_n,
            vr,
            vt,
            plith,
            top_Vnodes,
            bot_Vnodes,
            left_Vnodes,
            right_Vnodes,
            taurr_n,
            tautt_n,
            taurt_n,
            experiment,
            particle_rho_projection,
            particle_eta_projection,
            ls_rho_a,
            ls_eta_a,
            export_strainrate_tensor_components,
            export_devstress_tensor_components,
            export_stress_tensor_components,
            output_folder,
        )

        print("output solution to vtu file: ................. %.3f s" % (clock.time() - start))
        timings[10] += clock.time() - start

    ###############################################################################################
    # @@ output particles to vtu file
    ###############################################################################################
    start = clock.time()

    if istep % every_swarm_vtu == 0 or istep == nstep - 1:
        output_swarm_to_vtu(
            solve_Stokes,
            use_melting,
            TKelvin,
            istep,
            geometry,
            nparticle,
            nmat,
            solve_T,
            vel_scale,
            material_names,
            swarm_active,
            swarm_id,
            swarm_x,
            swarm_z,
            swarm_u,
            swarm_w,
            swarm_wf,
            swarm_rho,
            swarm_eta,
            swarm_r,
            swarm_t,
            swarm_p,
            swarm_paint,
            swarm_exx,
            swarm_ezz,
            swarm_exz,
            swarm_T,
            swarm_iel,
            swarm_hcond,
            swarm_hcapa,
            swarm_alpha,
            swarm_mechanism,
            swarm_rad,
            swarm_theta,
            swarm_strain,
            swarm_F,
            swarm_sst,
            output_folder,
        )

        print("output particles to vtu file: ................ %.3f s" % (clock.time() - start))
        timings[20] += clock.time() - start

    ###############################################################################################
    # @@ output particles to png file
    ###############################################################################################
    start = clock.time()

    if istep>0 and istep % every_swarm_png == 0:
        output_swarm_to_png(
            Lx,
            Lz,
            solve_Stokes,
            solve_T,
            istep,
            geometry,
            nparticle,
            nmat,
            material_names,
            swarm_active,
            swarm_x,
            swarm_z,
            swarm_u,
            swarm_w,
            swarm_wf,
            swarm_rho,
            swarm_eta,
            swarm_r,
            swarm_t,
            swarm_p,
            swarm_paint,
            swarm_exx,
            swarm_ezz,
            swarm_exz,
            swarm_T,
            swarm_iel,
            swarm_hcond,
            swarm_hcapa,
            swarm_alpha,
            swarm_rad,
            swarm_theta,
            swarm_strain,
            swarm_F,
            swarm_sst,
            output_folder,
        )

        print("output particles to png file: ................ %.3f s" % (clock.time() - start))
        timings[35] += clock.time() - start

    ###############################################################################################
    # @@ output particles to ascii file
    ###############################################################################################
    start = clock.time()

    if istep % every_swarm_ascii == 0:
        output_swarm_to_ascii(
            Lx,
            Lz,
            solve_Stokes,
            solve_T,
            istep,
            geometry,
            nparticle,
            swarm_active,
            swarm_x,
            swarm_z,
            swarm_u,
            swarm_w,
            swarm_wf,
            swarm_rho,
            swarm_eta,
            swarm_r,
            swarm_t,
            swarm_p,
            swarm_paint,
            swarm_exx,
            swarm_ezz,
            swarm_exz,
            swarm_T,
            swarm_iel,
            swarm_hcond,
            swarm_hcapa,
            swarm_alpha,
            swarm_rad,
            swarm_theta,
            swarm_strain,
            swarm_F,
            swarm_sst,
            output_folder,
        )

        print("output particles to ascii file: .............. %.3f s" % (clock.time() - start))
        timings[36] += clock.time() - start

    ###############################################################################################
    # @@ output quadrature points to vtu file
    ###############################################################################################
    start = clock.time()

    if istep % every_quadpoints_vtu == 0 or istep == nstep - 1:
        output_quadpoints_to_vtu(
            istep,
            nel,
            nq_per_element,
            nq,
            solve_T,
            xq,
            zq,
            rhoq,
            etaq,
            Tq,
            hcondq,
            hcapaq,
            dpdxq,
            dpdzq,
            gx_q,
            gz_q,
            output_folder,
        )

        print("output quad pts to vtu file: ................. %.3f s" % (clock.time() - start))
        timings[22] += clock.time() - start

    ###############################################################################################
    # @@ output solution to png file
    ###############################################################################################
    start = clock.time()

    if (istep % every_solution_png == 0 or istep == nstep - 1) and not inside_nonlinear_iterations:
        output_solution_to_png(
            geometry,
            solve_Stokes,
            solve_T,
            istep,
            vel_scale,
            vel_unit,
            TKelvin,
            nelx,
            nelz,
            Lx,
            Lz,
            x_V,
            z_V,
            u,
            w,
            q,
            T,
            eta_n,
            rho_n,
            exx_n,
            ezz_n,
            exz_n,
            e_n,
            divv_n,
            qx_n,
            qz_n,
            output_folder,
        )

        print("output solution to png file: ................. %.3f s" % (clock.time() - start))
        timings[34] += clock.time() - start

    ###############################################################################################
    # @@ compute avrg temperature, viscosity, velocity profiles
    # not the most elegant but works
    ###############################################################################################
    start = clock.time()

    if istep % every_solution_vtu == 0 or istep == nstep - 1:
        T_profile, vel_profile, eta_profile, q_profile, rho_profile, coord_profile = compute_avrg_profiles(
            geometry, nnx, nnz, T, rho_n, eta_n, u, w, q, z_V, rad_V
        )

        np.savetxt(
            output_folder+"/profiles/avrg_profile_q_" + str(istep) + ".ascii",
            np.array([coord_profile, q_profile]).T,
            header="#z,T",
        )
        np.savetxt(
            output_folder+"/profiles/avrg_profile_T_" + str(istep) + ".ascii",
            np.array([coord_profile, T_profile]).T,
            header="#z,T",
        )
        np.savetxt(
            output_folder+"/profiles/avrg_profile_eta_" + str(istep) + ".ascii",
            np.array([coord_profile, eta_profile]).T,
            header="#z,eta",
        )
        np.savetxt(
            output_folder+"/profiles/avrg_profile_rho_" + str(istep) + ".ascii",
            np.array([coord_profile, rho_profile]).T,
            header="#z,rho",
        )
        np.savetxt(
            output_folder+"/profiles/avrg_profile_vel_" + str(istep) + ".ascii",
            np.array([coord_profile, vel_profile]).T,
            header="#z,vel",
        )

    print("compute avrg profile: ........................ %.3f s" % (clock.time() - start))
    timings[9] += clock.time() - start

    ###############################################################################################
    # @@ compute gravitational field above domain
    # xs[npts],ys: coordinates of satellite
    # gxI,gzI,gnormI: gravity from internal density distribution
    # gxDTt,gzDTt,gnormDTt: gravity from dynamic topography at top
    # gxDTb,gzDTb,gnormDTb: gravity from dynamic topography at bottom
    ###############################################################################################
    start = clock.time()

    if gravity_npts > 0 and not inside_nonlinear_iterations:
        if istep == 0:
            xs = np.zeros(gravity_npts, dtype=np.float64)
            zs = np.zeros(gravity_npts, dtype=np.float64)
            gxI = np.zeros((gravity_npts, nstep), dtype=np.float64)
            gzI = np.zeros((gravity_npts, nstep), dtype=np.float64)
            grI = np.zeros((gravity_npts, nstep), dtype=np.float64)
            gtI = np.zeros((gravity_npts, nstep), dtype=np.float64)
            gnormI = np.zeros((gravity_npts, nstep), dtype=np.float64)
            gnormI_rate = np.zeros((gravity_npts, nstep), dtype=np.float64)
            gxDTt = np.zeros((gravity_npts, nstep), dtype=np.float64)
            gzDTt = np.zeros((gravity_npts, nstep), dtype=np.float64)
            gnormDTt = np.zeros((gravity_npts, nstep), dtype=np.float64)
            gnormDTt_rate = np.zeros((gravity_npts, nstep), dtype=np.float64)
            gxDTb = np.zeros((gravity_npts, nstep), dtype=np.float64)
            gzDTb = np.zeros((gravity_npts, nstep), dtype=np.float64)
            gnormDTb = np.zeros((gravity_npts, nstep), dtype=np.float64)
            gnormDTb_rate = np.zeros((gravity_npts, nstep), dtype=np.float64)

        match geometry:
            case "box":
                for i in range(0, gravity_npts):
                    xs[i] = i * Lx / (gravity_npts - 1)
                    zs[i] = Lz + gravity_height
                    gxI[i, istep], gzI[i, istep], gnormI[i, istep] = compute_gravity_at_point(
                        xs[i], zs[i], nel, x_e, z_e, rho_e, area, gravity_rho_ref
                    )

                    gxDTt[i, istep], gzDTt[i, istep], gnormDTt[i, istep] = compute_gravity_fromDT_at_point(
                        xs[i],
                        zs[i],
                        Lz,
                        nelx,
                        x_V[top_Vnodes],
                        rho_n[top_Vnodes],
                        dyn_topo_top,
                        rho_DT_top,
                    )

                    gxDTb[i, istep], gzDTb[i, istep], gnormDTb[i, istep] = compute_gravity_fromDT_at_point(
                        xs[i],
                        zs[i],
                        0,
                        nelx,
                        x_V[bot_Vnodes],
                        rho_n[bot_Vnodes],
                        dyn_topo_bot,
                        rho_DT_bot,
                    )

                np.savetxt(
                    output_folder+"/gravityI_" + str(istep) + ".ascii",
                    np.array([xs, zs, gnormI[:, istep], gxI[:, istep], gzI[:, istep]]).T,
                    header="#x,z,g,gx,gz",
                )
                np.savetxt(
                    output_folder+"/gravityDTt_" + str(istep) + ".ascii",
                    np.array([xs, zs, gnormDTt[:, istep], gxDTt[:, istep], gzDTt[:, istep]]).T,
                    header="#x,z,g,gx,gz",
                )
                np.savetxt(
                    output_folder+"/gravityDTb_" + str(istep) + ".ascii",
                    np.array([xs, zs, gnormDTb[:, istep], gxDTb[:, istep], gzDTb[:, istep]]).T,
                    header="#x,z,g,gx,gz",
                )

            case "quarter" | "half":
                for i in range(0, gravity_npts):
                    xs[i] = (Router + gravity_height) * np.cos(i / (gravity_npts - 1) * opening_angle + theta_min)
                    zs[i] = (Router + gravity_height) * np.sin(i / (gravity_npts - 1) * opening_angle + theta_min)
                    gxI[i, istep], gzI[i, istep], gnormI[i, istep] = compute_gravity_at_point(
                        xs[i], zs[i], nel, x_e, z_e, rho_e, area, gravity_rho_ref
                    )

                rads = np.sqrt(xs**2 + zs**2)
                thetas = np.pi / 2 - np.arctan2(xs, zs)
                grI[:, istep] = gxI[:, istep] * np.cos(thetas) + gzI[:, istep] * np.sin(thetas)
                gtI[:, istep] = -gxI[:, istep] * np.sin(thetas) + gzI[:, istep] * np.cos(thetas)

                np.savetxt(
                    output_folder+"/gravityI_" + str(istep) + ".ascii",
                    np.array([rads, thetas, gnormI[:, istep], grI[:, istep], gtI[:, istep]]).T,
                    header="#r,theta,g,gx,gz",
                )

            case _:
                print("gravity calculations not available for this geometry")

        if istep > 0:
            gnormI_rate[:, istep] = (gnormI[:, istep] - gnormI[:, istep - 1]) / dt
            gnormDTt_rate[:, istep] = (gnormDTt[:, istep] - gnormDTt[:, istep - 1]) / dt
            gnormDTb_rate[:, istep] = (gnormDTb[:, istep] - gnormDTb[:, istep - 1]) / dt
            if geometry == "box":
                np.savetxt(
                    output_folder+"/gravityI_rate_" + str(istep) + ".ascii",
                    np.array([xs, gnormI_rate[:, istep] / mGal * year]).T,
                    header="#x,g",
                )
                np.savetxt(
                    output_folder+"/gravityDTt_rate_" + str(istep) + ".ascii",
                    np.array([xs, gnormDTt_rate[:, istep] / mGal * year]).T,
                    header="#x,g",
                )
                np.savetxt(
                    output_folder+"/gravityDTb_rate_" + str(istep) + ".ascii",
                    np.array([xs, gnormDTb_rate[:, istep] / mGal * year]).T,
                    header="#x,g",
                )
            if geometry == "quarter" or geometry == "half":
                np.savetxt(
                    output_folder+"/gravityI_rate_" + str(istep) + ".ascii",
                    np.array([thetas, gnormI_rate[:, istep] / mGal * year]).T,
                    header="#theta,g",
                )

    print("compute gravity: ............................. %.3f s" % (clock.time() - start))
    timings[23] += clock.time() - start


    ###############################################################################################

    if istep % 10 == 0 or istep == nstep - 1 or geo_time > end_time:
        duration = clock.time() - topstart
        print_timings(iloop,timings,duration)

    dtimings = timings - timings_mem
    dtimings[0] = istep+iter_nl/100
    dtimings.tofile(timings_file, sep=" ", format="%e")
    timings_file.write(" \n")
    timings_mem[:] = timings[:]

    ###########################################################################

    if geometry == "box" and nsamplepoints > 0:
        sample_solution_box(
            nn_V,
            x_V,
            z_V,
            u,
            w,
            q,
            T,
            nsamplepoints,
            xsamplepoints,
            zsamplepoints,
            Lx,
            Lz,
            nelx,
            nelz,
        )


    ###########################################################################

    if geo_time > end_time:
        print("***** end time reached *****")
        break

    if istep==nstep-1:
        print("***** nb of steps reached *****")
        break

    if inside_nonlinear_iterations:
       iter_nl+=1
    else:
       geo_time += dt 
       iter_nl=0
       istep+=1
       print("NL: resetting iter_nl=0")
       print("NL: increasing istep: istep=",istep)

# end for iloop
# @@ --------------------- end time stepping loop ------------------------------

pvd_solution_file.write("  </Collection>\n")
pvd_solution_file.write("</VTKFile>\n")
pvd_swarm_file.write("  </Collection>\n")
pvd_swarm_file.write("</VTKFile>\n")

###############################################################################
# output horizontal and vertical profiles
###############################################################################

output_final_profiles(
    x_V,
    z_V,
    u,
    w,
    q,
    T,
    rho_n,
    eta_n,
    middleV_nodes,
    middleH_nodes,
    x_e,
    z_e,
    p_e,
    rho_e,
    eta_e,
    middleV_element,
    middleH_element,
    output_folder,
)

###############################################################################
###############################################################################

output_test_results(m_V,x_V,z_V,Nfem_V,\
                    m_P,x_P,z_P,Nfem_P,\
                    m_T,x_T,z_T,Nfem_T,\
                    nelx,nelz,vrms,T_avrg,output_folder)

###############################################################################
# close files
###############################################################################


vstats_file.close()
pstats_file.close()
srstats_file.close()
vrms_file.close()
dt_file.close()
TM_file.close()
EK_file.close()
WAG_file.close()
T_avrg_file.close()
eta_avrg_file.close()
delta_file.close()
etaq_file.close()
etae_file.close()
etan_file.close()
conv_file.close()
if solve_T:
    Nu_file.close()
    corner_q_file.close()
    Tstats_file.close()
    avrg_T_bot_file.close()
    avrg_T_top_file.close()
    avrg_dTdz_bot_file.close()
    avrg_dTdz_top_file.close()

###############################################################################

print("-----------------------------")
print("total compute time: %.1f s" % (duration))
print("sum timings: %.1f s" % (np.sum(timings)))
print("-----------------------------")

###############################################################################
