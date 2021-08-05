import numpy as np
from scipy.special import kv

def RasSoln(test_list, s, soln):
    s = np.exp(s)
    
    om = test_list[:,1] # Angular frequency (rad/s) 
    Q_0 = test_list[:,2] # Max pumping rate (m^3/s)
    r = test_list[:,4] # Radial distance (m)

    num_obs = r.size
    phasor_mod = np.array((0,))
    
    for i in range(num_obs):
        print(i)
        if soln == 'leaky':
            arg = np.sqrt(((1j * r[i]**2 * om[i]) / s[0]) + (r[i]**2 / s[2]))

        elif soln == 'confined':
            arg = np.sqrt((1j * r[i]**2 * om[i]) / s[0])
        
        phasor_mod = Q_0 / (2 * np.pi * s[1]) * kv(0, arg)
        breakpoint()
    
#    y_mod = np.zeros((2*num_obs,1))
#    y_mod[0:2:-2] = phasor_mod.real
#    y_mod[1:2:-1] = phasor_mod.imag

    return phasor_mod
    
def jacobian(s, delta, test_list, soln):
     num_param = np.size(s)
     num_obs = np.size(test_list[:,0])
     
     J = np.zeros((2*num_obs, num_param))
     coeffs_base = RasSoln(test_list, s, soln)
     for i in s:
         sj = s
         sj[i] = i + delta[i]

         coeffs_mod = RasSoln(test_list, sj, soln)



# def lev_marq():
import numpy as np
P = 10
omega = (2 * np.pi) / P
Q_max = 5e-3
Q_phase = 0
r = 10
soln = 'confined'

test_list = np.matrix([[P, omega, Q_max, Q_phase, r, 0.13, -0.5], [90, (2*np.pi)/90, Q_max, Q_phase, 15, 0.13, -0.5]])
s_opt = np.array([[2], [-8]])
print(test_list)
y_mod = RasSoln(test_list, s_opt, soln)
print(y_mod)