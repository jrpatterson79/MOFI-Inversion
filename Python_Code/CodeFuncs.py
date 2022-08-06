import numpy as np
from scipy.special import kv
from scipy import optimize

# Fully confined analytical model from Rasmussen et al., (2003)
def RasSoln(P, Q_max, r, s_ln, soln):
    s = np.exp(s_ln)
    omega = np.reshape(((2 * np.pi) / P), (r.size,1))
    
    # Fully confined analytical model from Rasmussen et al., (2003)
    if soln == 'confined':
        arg = np.sqrt((1j * r**2 * omega) / s[0])
    
    # Leaky analytical model from Rasmussen et al., (2003)
    elif soln == 'leaky':
        B_sq = s[1] / s[2]
        arg = np.sqrt(((1j * r**2 * omega) / s[0]) + (r**2 / B_sq))        

    else:
        print('Error: Pick a valid analytical model')

    # Complex valued phasor 
    phasor_mod = (Q_max / (2 * np.pi * s[1])) * kv(0, arg)

    # Fourier coefficient array used for inversion. Order is [real, imag] for each 
    y_mod = np.zeros([r.size*2,1])
    y_mod[0:-1:2] = phasor_mod.real
    y_mod[1::2] = phasor_mod.imag
    return y_mod

def jacobian(P, Q_max, r, s_ln, delta, soln):
    num_param = s_ln.size
    num_data = r.size*2
     
    J = np.zeros((num_data, num_param))
    coeffs_base = RasSoln(P, Q_max, r, s_ln, soln)
    
    sj = s_ln + np.diagflat(delta)
    for i in range(0, num_param):
        coeffs_mod = RasSoln(P, Q_max, r, sj[:,i], soln)
        J[:,i] = ((coeffs_mod - coeffs_base) / delta[i]).T
    return(J)

def LevMarq(P, Q_max, r, s_ln, delta, lam, y, soln):
    # Closure Criteria
    max_linevals = 100
    obj_close = 1e-6
    s_close = 1e-6
    max_iter = 50

    #Initiate Parameter Estimation
    s_curr = s_ln
    lam = lam_init

    num_fxn_evals = 0
    num_iter = 0

    # Anonymous Functions
    h = lambda s: RasSoln(P, Q_max, r, s, soln)
    obj_fxn = lambda s: (1/2) * np.linalg.multi_dot([(h(s)-y).T, R_inv, (h(s)-y)])

    while iter < iter_max:
        print('Iteration', iter)
        print('Lambda = ', lam)
        y_curr = h(s_curr)

        J = jacobian(P, Q_max, r, s_curr, delta, soln)
        lhs = -np.dot(J.T, J) + (lam * np.eye(s_ln.size)) 
        rhs = np.dot(-J.T, (y-y_curr))
        step = np.linalg.lstsq(lhs, rhs)
        step_obj = lambda alpha : obj_fxn(s_curr + (alpha * step))
        alpha_best = optimize.fmin(step_obj, 0.5)

        s_new = s_curr + (step * alpha_best)
        s_change = max(abs(s_new - s_curr) / s_curr)

        obj_curr = obj_fxn(s_curr)
        obj_new = obj_fxn(s_new)
        obj_change = abs(obj_new - obj_curr) / obj_curr

        if obj_change <= obj_close and s_change <= s_close:
            return s_curr

        else:
            if obj_new < obj_curr:
                s_curr = s_new;
                obj_curr = obj_new;
                if lam <= 1e-12:
                    lam = 1e-12
                else:
                    lam = lam * 1e-1
                iter = iter + 1
                
            else:
                if lam >= 1e10:
                    lam = 1e10

                else:
                    lam = lam * 1e1

                iter = iter + 1
            

    s_hat = s_curr
    out_flag = 0
    