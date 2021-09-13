%Jeremy Patterson
%FSR^2

%This function file utilizes the gradient based nonlinear
%Levenberg-Marquardt optimization algorithm.

function varargout = Lev_Marq(test_list, s, y, R_inv, lambda_init, delta, soln)
% Lev_Marq: This function conducts non-linear gradient inversion using the Levenberg-Marquardt algorithm. The inversion uses analytical solutions developed by Rasmussen et al., % (2003) as the forward model to determine optimal flow parameters. 

% Forward Model
h = @(s) RasSoln(test_list, s, soln);
% Objective Function
obj_func = @(s) (1/2) * (y - h(s))' * R_inv * (y - h(s));

%Closure Criteria
max_linevals = 100;
obj_close = 1e-6;
s_close = 1e-6;
max_iter = 75;

% Initiate parameter estimation
s_curr = s;
lambda = lambda_init;

num_func_evals = 0;
iter = 0

num_data = numel(y);
num_param = numel(s_curr);

while iter < max_iter
    
    % Calculate the modeled data with current and new parameters
    y_curr = h(s_curr);    
    
    %Calculate the jacob of the forward model
    [J] = jacob(s_curr, delta, test_list, soln);
    
    % Calculate the Gauss-Newton or Levenberg-Marquardt step
    step = -(J' * J + lambda * eye(num_param,num_param))\(-J'*(y_curr - y));
    
    % Do the line search along the Gauss-newton step direction
    step_obj = @(alpha) obj_func(s_curr + (alpha .* step));
    options = optimset('Display', 'iter', 'MaxFunEvals', max_linevals);
    alpha_best = fminsearch(step_obj, 0.5, options);    
    
    % Calculate relative parameter change
    s_new = s_curr + (alpha_best .* step);
    s_change = max(abs((s_new - s_curr) ./ s_curr))
    
    % Calculate relative objective function change
    obj_curr = log10(obj_func(s_curr));
    obj_new = log10(obj_func(s_new));
    obj_change = abs((obj_curr - obj_new)./obj_curr)
    
    if (obj_change <= obj_close) && (s_change <= s_close)
        s_hat = s_curr;
        out_flag = 1;
        varargout = {s_hat, s_update, out_flag};
        return;
    else
               
        if obj_new < obj_curr
            s_update(iter+1,:) = s_curr;
            
            s_curr = s_new;
            obj_curr = obj_new;
            
            if lambda <= 1e-12
                lambda = 1e-12
            else
                lambda = lambda * 1e-1
            end
            iter = iter + 1
            
        else
            s_update(iter+1,:) = s_curr;
            
            if lambda >= 1e10
                lambda = 1e10
            else
                lambda = lambda * 1e1
            end
            iter = iter + 1
        end
    end
end
s_hat = s_curr;
out_flag = 0;
varargout = {s_hat, s_update, out_flag};
warning('Max iterations exceeded')