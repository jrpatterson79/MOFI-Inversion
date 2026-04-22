include("func_file.jl")
# Define aquifer Geometry
r = 10; # Radial distance from pumping well [m]

# True aquifer Parameters
D_true = 2;               # Hydraulic Diffusivity (ln(m^2/s)) 
T_true = -8;              # Transmissivity (ln(m^2/s))
S_true = T_true - D_true; # Storativity (ln(-))

# Pumping Amplitude (m^3/s)
Q_max = 7e-5;
P = 30;
delta = [0.1; 0.1];

phasor = steady_periodic_func(P, Q_max, r, [T_true; S_true])
J = jacobian([T_true; S_true], delta, P, Q_max, r)