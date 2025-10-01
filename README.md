# meeuuw
Mantle modelling Early Earth Utrecht University Work-in-progress

Code description:
- FEM
- Q_2xQ_1 finite element pair for velocity
- Q_2 finite element for temperature
- 2d Cartesian geometry
- flexible nonlinear viscous rheology 
- particle-in-cell technique
- Runge-Kutta in space: 1st, 2nd, 4th order
- export to vtu file (nodes, swarm, and quadrature points)
- direct solver for both linear systems
- Crank-Nicolson time scheme for T equation

Available experiments:
- experiment 0: Blankenbach et al, GJI, 1989. Mantle convection benchmark.
- experiment 1: van Keken et al, JGR, 1997. Rayleigh-Taylor instability.
- experiment 2: 
- experiment 3: Tosi et al, G3, 2015. Viscoplastic thermal convection benchmark.

to do:
- more accurate heat flux calculations (CBF?)
- more accurate whole domain velocity gradient method
- SUPG and/or Lenardic & Kaula filter
- nonlinear iterations
- cvi for Q2 ? 
- make hcond,hcapa nodal quantities
- introduce tfinal
0 implement EBA, ALA, TALA, ICA? 

Nomenclature

- nel: number of elements
- nn_V: number of velocity nodes
- nn_P: number of pressure nodes
- Nfem_V: number of velocity degrees of freedom
- Nfem_P: number of pressure degrees of freedom
