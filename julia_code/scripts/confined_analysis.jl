# Fully confined aquifer single and multiple frequency analysis presented in:
# Patterson, J.R.; Cardiff, M.A.; Aquifer Characterization and Uncertainty in Multi-Frequency Oscillatory Flow Tests: Approach and Insights; Groundwater; 2021; doi:10.1111/gwat.13134
# Code by Jeremy Patterson 12/2020, Last Updated 12/2021


# Seed the random number generators
randn('state', 0); # Data error noise seed

# Define aquifer Geometry
r = 10; # Radial distance from pumping well [m]

# True aquifer Parameters
D_true = 2;               # Hydraulic Diffusivity (ln(m^2/s)) 
T_true = -8;              # Transmissivity (ln(m^2/s))
S_true = T_true - D_true; # Storativity (ln(-))

# Define Parameter Surface
T_vec = (T_true-10:5e-2:T_true+10);
S_vec = (S_true-10:5e-2:S_true+10);
[T, S] = meshgrid(T_vec, S_vec);

# Pumping Amplitude (m^3/s)
Q_max = 7e-5;

# Sampling Frequency
dt = 1/8;
data_err = 1e-4; # Data error variance (Assumes 1 cm data measurement error)

# LM Inversion Inital Parameters
delta = [0.1 0.1];
lambda = 1e1;
s_init = [T_true-1; S_true-1];
# s_init = [-15; -14]; # Drives inversion to local minimum