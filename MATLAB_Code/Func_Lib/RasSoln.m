function [y_mod] = RasSoln(test_list, s, soln)

s = exp(s);
D = s(1) / s(2);

r = test_list(:,4);
num_obs = numel(r);

omega = test_list(:,2);

Q_max = test_list(:,3);
phasor_mod = zeros(num_obs, 1);

for j = 1 : num_obs
    
    if strcmp(soln, 'leaky') == 1
        B_sq = s(1) / s(3);
        arg = sqrt(((1i * r(j)^2 * omega(j)) / D) +...
                    (r(j)^2 / B_sq));
        
    elseif strcmp(soln, 'confined') == 1
        arg = sqrt((1i * r(j)^2 * omega(j)) / D);
    end
    
    phasor_mod(j) =  Q_max(j) / (2 * pi * s(1)) * besselk(0, arg);
end
y_mod = zeros(2*num_obs,1);
y_mod(1:2:end-1) = real(phasor_mod);
y_mod(2:2:end) = imag(phasor_mod);
