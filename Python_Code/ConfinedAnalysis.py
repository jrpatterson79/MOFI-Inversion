import numpy as np
import random as rdm

# Seed the random number generators
np.random.seed(1)
np.random.randn(100,1)

# Specify Model Type
soln = 'confined';

# Aquifer Geometry
r = 10;

# Aquifer Parameters
D_true = 2;
T_true = -8;
S_true = T_true - D_true;

# Define Parameter Surface
T_vec = np.arange(T_true-10, T_true+10.05, 5e-2)
S_vec = np.arange(S_true-10, S_true+10.05, 5e-2)
T,S = np.meshgrid(T_vec, S_vec)

# Pumping Amplitude (m^3/s)
Q_max = 7e-5;

# Sampling Frequency
dt = 1/8;

# Data error variance (Assumes 1 cm data measurement error)
data_err = np.sqrt(1e-4); 

# LM Inversion Inital Parameters
delta = np.array([[0.1],[0.1]])
lam_init = 1e1;
s_init = np.array([[T_true-2],[S_true-2]])
# s_init = np.array([[-15], [-14]]) % Drives inversion to local minimum

P = 90
omega = (2 * np.pi) / P
