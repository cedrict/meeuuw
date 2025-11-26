CFLnb=0.5
tol_ss=-1e-8
pressure_normalisation='surface'
every_Nu=1000000
every_solution_vtu=1
every_swarm_vtu=1
every_quadpoints_vtu=1000000
end_time=0.
formulation='BA'
vel_scale=1 ; vel_unit=' '
time_scale=1 ; time_unit=' '
p_scale=1 ; p_unit=' '
solve_T=False
solve_Stokes=True
method_nodal_strain_rate=1
remove_rho_profile=False
top_free_slip=False
bot_free_slip=False
nstep=1
nqperdim=3
straighten_edges=False
use_elemental_rho=False
use_elemental_eta=False
mapping='Q2'
averaging='geometric'
use_melting=False

#######################################
# debug pparameters 
#######################################

debug_ascii=False
debug_nan=False

#######################################
# geometry parameters
#######################################

axisymmetric=False
TKelvin=0
geometry='box'
Lx=1
Lz=1
Rinner=0.55
Router=1

#######################################
# particle in cell parameters
#######################################

particle_distribution=0 # 0: random, 1: reg, 2: Poisson Disc, 3: pseudo-random
RKorder=2
nparticle_per_dim=5

#######################################
# gravity & dyn topo parameters
#######################################

gravity_npts=0 
gravity_rho_ref=0
gravity_height=0
rho_DT_top=1e-10
rho_DT_bot=1e-10

