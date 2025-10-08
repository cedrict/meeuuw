
cm=0.01
year=365.25*3600*24


Lx=11600e3
Ly=2900e3
gy=-9.81
eta_ref=1e22
solve_T=True
pressure_normalisation='surface'
p_scale=1e6 ; p_unit="MPa"
vel_scale=cm/year ; vel_unit='cm/yr'
time_scale=year ; time_unit='yr'
every_Nu=100000
TKelvin=273.15
end_time=1e9*year
Tbottom=3000+TKelvin
Ttop=0+TKelvin

rho0=3300
alphaT=2e-5
T0=TKelvin
hcond0=5
hcapa0=1250
eta0=2e22

print(hcond0/hcapa0/rho0 )
print( (Tbottom-Ttop)*rho0*abs(gy)*alphaT*Ly**3 / eta0 / (hcond0/hcapa0/rho0))

nelx=128
nely=32
nstep=100
CFLnb=0.5           
