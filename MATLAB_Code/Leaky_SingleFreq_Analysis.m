% Single-Frequency Analysis Leaky Conceptual Model (Patterson and Cardiff, 2022)
% Code developed by Jeremy Patterson
% Created Dec 2020; Updated May 2021

% This code generates the single-frequency analysis seen in the section "Leaky Aquifer System"
% Patterson, J. R., & Cardiff, M. (2022). Aquifer Characterization and Uncertainty in Multi-Frequency Oscillatory Flow Tests: Approach and Insights. Groundwater, 60(2),
% 180–191. https://doi.org/10.1111/gwat.13134

%% Clean Environment
close all; clear; clc

%% Specify Directory
% Specify the directory location of the folder Func_Lib, which contains the needed function files to execute this code
addpath('/.../.../') 

%% Create Synthetic Data
% Seed the random number generators
randn('state', 0);   % Data noise
normrnd('state', 0); % Seed for multi-freq stochastic sampling

% Specify model type
soln = 'leaky';

% Aquifer Geometry
r = 10;

% Aquifer Parameters (in ln space)
D_true = 2;
T_true = -8;
S_true = T_true-D_true;
L_true = -21;

% Define Parameter Surface
T_vec = (T_true-5:1e-2:T_true+5);
S_vec = (S_true-5:1e-2:S_true+5);
L_vec = (L_true-2:1e-2:L_true+10);

[T, S, L] = meshgrid(T_vec, S_vec, L_vec);
param = [reshape(T, [], 1) reshape(S, [], 1) reshape(L, [], 1)];

% Stimulation Period Vector
P = 7200;
omega = (2*pi) ./ P;
Q_max = 7e-5;

% Generate Test List
synth_data = [];
for i = 1 : numel(r)
    for j = 1 : numel(P)
     synth_data = [synth_data; ...
      P(j) omega(j) Q_max r(i)]; 
    end
end
num_obs = numel(synth_data(:,1));

% Generate true phasors w/o noise
y_mod = @(s) RasSoln(synth_data, s, soln);
y_true = y_mod([T_true; S_true; L_true]);

%% Parameter Space Search
dt = 1/8; % Sampling frequency
data_err = 1e-4; % Observation signal noise variance (Assumes 1 cm data measurement error)

% Generates noisy signal using Gaussian noise under i.i.d. assumption
phasor = [y_true(1:2:end-1) y_true(2:2:end)];
for i = 1 : num_obs
    t = [0 : dt : 5.*synth_data(i,1)]';
    signal = (phasor(i,1) .* cos(synth_data(i,2) .* t)) +...
             (-phasor(i,2) .* sin(synth_data(i,2) .* t));
    noise = data_err .* randn(size(t));
    sig_noise = signal + noise;
    [data_cov{i}, phasor_noise(i,:)] = periodic_LS_fit(t, sig_noise, synth_data(i,1));
    y_stddev(:,i) = 1.96 * sqrt(diag(data_cov{i}));
end
y_noise = zeros(2*num_obs,1);

% Fourier coefficients with added noise. Odd elements are real
% components, even elements are imaginary components.
y_noise(1:2:end-1) = real(phasor_noise);
y_noise(2:2:end) = imag(phasor_noise);

y_sim = zeros(numel(param(:,1)), 2);
for k = 1 : numel(param(:,1))
    y_sim(k,:) = y_mod([param(k,1); param(k,2); param(k,3)]);
end
idx = find(abs(y_sim(:,1)-y_true(1))<=5e-4 & abs(y_sim(:,2)-y_true(2))<=5e-4);

%% Figures
p_idx = idx(1:10:end);

% Figure 6
figure(6)
clf
ax = gca;
plot3(param(p_idx,1), param(p_idx,2), param(p_idx,3), 'k^',...
      'MarkerSize', 6, 'MarkerFaceColor', [0.4660, 0.6740, 0.1880])
hold on
plot3(T_true, S_true, L_true, 'kd', 'LineWidth', 2,...
      'MarkerSize', 12, 'MarkerFaceColor', [0.8500, 0.3250, 0.0980])
plot3(param(p_idx,1), min(S_vec) * ones(numel(p_idx),1), param(p_idx, 3), '.', 'Color', [0.7 0.7 0.7], 'MarkerSize', 12)
plot3(param(p_idx,1), param(p_idx,2), -22*ones(numel(p_idx),1), '.', 'Color', [0.7 0.7 0.7], 'MarkerSize', 12)
plot3(max(T_vec) * ones(numel(p_idx),1), param(p_idx,2), param(p_idx,3), '.', 'Color', [0.7 0.7 0.7], 'MarkerSize', 12)
grid on
axis([min(T_vec) max(T_vec) min(S_vec) max(S_vec) -22 -14])
view(-135, 25)
xlabel('ln(T [m^2/s])')
ylabel('ln(S [-])')
zlabel('ln(L [s^{-1}])')
ax.FontSize = 18;
ax.YTick = [min(S_vec):2.5:max(S_vec)];
l = legend('s_{opt}', 's_{true}');
l.FontSize = 18;
set(gcf, 'Position', [100 100 900 600])