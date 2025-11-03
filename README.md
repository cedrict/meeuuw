# meeuuw
Mantle modelling Early Earth Utrecht University Work-in-progress

## Code description:
- FEM
- Q_2xQ_1 finite element pair for velocity-pressure
- Q_2 finite element for temperature
- 2d Cartesian and 2d annulus geometry
- flexible nonlinear viscous rheology 
- particle-in-cell technique
- Runge-Kutta in space: 1st, 2nd, 4th order
- export to vtu file (nodes, particle swarm, and quadrature points)
- direct solver for both linear systems
- Crank-Nicolson time scheme for T equation

## Available experiments:
- experiment 0: Blankenbach et al, GJI, 1989. Mantle convection benchmark.
- experiment 1: van Keken et al, JGR, 1997. Rayleigh-Taylor instability.
- experiment 2: Schmeling et al, PEPI, 2008. Newtonian subduction.
- experiment 3: Tosi et al, G3, 2015. Viscoplastic thermal convection benchmark.
- experiment 4: convection
- experiment 5: Trompert & Hansen, Nature 1998.
- experiment 6: Crameri et al, GJI 2012 (free surface benchmark)
- experiment 7: Delft workshop, October 2025.

## to do:
- more accurate heat flux calculations (CBF?)
- more accurate whole domain velocity gradient method
- SUPG and/or Lenardic & Kaula filter
- nonlinear iterations
- implement EBA, ALA, TALA, ICA? 
- solid phase transition
- melt generation and transport
- C matrix for compressible case
- compute sr or dev sr ?
- look at stsh04 (similar to trha98b)
- paint stripes adapt to aspect ratio
- what stress to use for DT?
- higher order mapping?
- heat production term

## Nomenclature

### Flags/parameters

- Lx,Ly: dimensions of the domain
- formulation: can take value 'BA', 'EBA', ...?
- averaging: 'arithmetic', 'geometric', 'harmonic' 
- geometry: 'box' or 'quarter'

### Finite elements

- u,v:  velocity components arrays
- T: temperature array
- p: pressure field (on Q1 mesh)
- q: pressure field (on Q2 mesh)
- nelx,nely: number of elements in each direction
- nel: number of elements
- nn_V: number of velocity nodes
- nn_P: number of pressure nodes
- ndof_V_el: number of V dofs per element (=18)
- Nfem_V: number of velocity degrees of freedom
- Nfem_P: number of pressure degrees of freedom
- Nfem_T: number of temperature degrees of freedom
- Nfem: total number of degrees of freedom (Stokes eqs) 
- m_V: number of velocity nodes per element (=9)
- m_P: number of pressure nodes per element (=4)
- m_T: number of temperature nodes per element (=9)
- hx,hy: size of an element
- x_V,y_V: coordinates arrays of velocity nodes
- x_P,y_P: coordinates arrays of pressure nodes
- icon_V: connectivity array for velocity nodes
- icon_P: connectivity array for pressure nodes
- bc_fix_V, bc_val_V: boundary conditions arrays for velocity
- bc_fix_T, bc_val_T: boundary conditions arrays for temperature
- N_V, dNdr_V, dNds_V, dNdx_V, dNdy_V: velocity basis functions and derivatives
- N_P: pressure basis functions
- exx_n,eyy_n,exy_n: nodal components of strain rate tensor
- vrms: root means square velocity
- qx_nodal, qy_nodal: nodal heat flux component
- II_V, JJ_V, VV_V: arrays to store Stokes FEM matrix
- II_T, JJ_T, VV_T: arrays to store energy FEM matrix
- r_V, s_V: arrays of size m_V containing red coords of V nodes
- rad_V, theta_V: polar coordinates of V nodes
- rad_P, theta_P: polar coordinates of P nodes

### Gauss quadrature 

- nqel: number of quadrature points per element 
- nq: total number of quadrature points in the domain
- xq(nel,nqel),yq(nel,nqel): coordinate arrays of quadrature points
- uq(nel,nqel),vq(nel,nqel): velocity components on quadrature points
- rhoq: density on quadrature points
- etaq: viscosity on quadrature points
- exxq, eyyq, exyq: strain rate components on quadrature points
- dpdxq, dpdyq: pressure gradient on quadrature points

### Particles

- particle_distribution: 
- swarm_X: field X carried by the swarm of particles
- RKorder: order of the Runge-Kutta algorithm
- nparticle_per_dim: number of articles per dimension
- nparticle: number of particles in the domain

---

## Code structure

- quadrature rule points and weights
- open output files & write headers
- build velocity nodes coordinates 
- connectivity for velocity nodes
- build pressure grid 
- build pressure connectivity array 
- define velocity boundary conditions
- define temperature boundary conditions
- initial temperature. T is a vector of float64 of size nn_V
- compute area of elements / sanity check
- precompute basis functions values at quadrature points
- precompute basis functions values at V nodes
- compute coordinates of quadrature points 
- compute gravity vector at quadrature points 
- compute gravity on mesh points
- compute array for assembly
- fill II_V,JJ_V arrays for Stokes matrix
- fill II_T,JJ_T arrays for temperature matrix
- particle coordinates setup
- particle paint
- particle layout
- --------------------- time stepping loop ----------------------------------
    - interpolate strain rate on particles
    - interpolate temperature on particles
    - evaluate density and viscosity on particles (and hcond, hcapa, hprod)
    - project particle properties on elements 
    - project particle properties on V nodes
    - project nodal values onto quadrature points
    - split solution into separate u,v,p velocity arrays
    - compute timestep
    - normalise pressure: simple approach to have avrg p = 0 (volume or surface)
    - project Q1 pressure onto Q2 (vel,T) mesh
    - project velocity on quadrature points
    - build temperature matrix
    - solve temperature system
    - compute vrms 
    - compute nodal heat flux 
    - compute heat flux and Nusselt at top and bottom
    - compute temperature profile
    - compute nodal strainrate
    - compute stress tensor components
    - compute dynamic topography at bottom and surface topo 
    - compute nodal pressure gradient 
    - advect particles
    - locate particles and compute reduced coordinates
    - export min/max coordinates of each material in one single file
    - generate/write in pvd files
    - export solution to vtu file
    - export particles to vtu file
    - export quadrature points to vtu file
    - compute gravitational field above domain 
    - assess steady state
- --------------------- end time stepping loop ------------------------------
