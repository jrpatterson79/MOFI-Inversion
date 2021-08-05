% FSR^2
% Jacobian approximation
% December 2020
% Jeremy Patterson

%This function calculates the parameter sensitivity matrix (i.e. Jacobian) of the finite difference based heat
%equation forward model.
%
%     Inputs
%
%
%     Outputs
%
%   [J] numdata x numparam sensitivity matrix 


function [J] = jacob(p, delta, test_list, soln)

% Modeled temperature value with unchanged parameters
[coeffs_base] = RasSoln(test_list, p, soln);
    
for i = 1 : numel(p)
    pj = p; 
    pj(i) = p(i) + delta(i);
    
    % Modeled Fourier coefficients with perturbed parameters
    [coeffs_mod] = RasSoln(test_list, pj, soln); 
    
    J(:,i) = (coeffs_mod - coeffs_base) ./ delta(i);
   
end


