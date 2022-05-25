function varargout = SigLenUnc(test_list, y, s, dt, t_max, data_err, lambda, delta, soln)



    num_obs = numel(test_list(:,1));
iter = 1;
t_total = 0;
signal_length = 2;

while t_total < t_max
    t_curr = 0;
    for k = 1 : num_obs
        t_new = signal_length * test_list(k,1);
        t_curr = t_curr + t_new;
        % Generates noisy signal using Gaussian noise under i.i.d. assumption
        t = [0 : dt : t_new]';
        sig = (y(k,1) .* cos(test_list(k,2) .* t)) +...
             (-y(k,2) .* sin(test_list(k,2) .* t));
        noise = data_err .* randn(size(t));
        sig_noise = sig + noise;
        
        % Least-squares approach to determine phasor coefficients
        [data_cov{k}, phasor_noise(k,:)] = periodic_LS_fit(t, sig_noise, test_list(k,1));             
    end
    
    % Inverse data error covariance matrix
    R_inv = inv(blkdiag(data_cov{1:end}));
    
    % Noisy phasor coefficients
    y_noise = zeros(2*num_obs,1);
    y_noise(1:2:end-1) = real(phasor_noise);
    y_noise(2:2:end) = imag(phasor_noise);
    
    % Conduct LM Inversion
    [s_hat(iter,:), ~, exit_flag(iter)] = Lev_Marq(test_list, s, y_noise, R_inv, lambda, delta, soln);

    % Linearized Uncertainty Analysis
    J = jacob(s_hat(iter,:), delta, test_list, soln);
    param_cov = inv(J' * R_inv * J);                     % Parameter Covariance Matrix
    param_stddev(iter,:) = 1.96 * sqrt(diag(param_cov)); % Parameter Standard Deviation
    
    t_save(iter,:) = t_curr;
    t_total = t_curr; 
    signal_length = signal_length + 1;
    iter = iter + 1;
end

if strcmp(soln, 'confined') == 1
    varargout = {t_save, param_stddev(:,1), param_stddev(:,2), s_hat};
elseif strcmp(soln, 'leaky') == 1
    varargout = {t_save, param_stddev(:,1), param_stddev(:,2), param_stddev(:,3), s_hat};
else
    error('Pick a valid analytical solution (confined or leaky)')
end
