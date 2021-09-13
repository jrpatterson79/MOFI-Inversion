% Fully confined aquifer single and multiple frequency analysis presented in:
% Patterson, J.R.; Cardiff, M.A.; Aquifer Characterization and Uncertainty in Multi-Frequency Oscillatory Flow Tests: Approach and Insights; Groundwater; 2021; doi:10.1111/gwat.13134
% Code by Jeremy Patterson 12/2020, Last Updated 09/2021


%% Clean Environment
close all; clear; clc

%% Specify Directory
% Set to automatically determine present working directory and add the subdirectory with all of the necessary function files. Change as necessary to suit your needs.
dir = pwd;
addpath([dir '/Func_Lib'])

% Seed the random number generators
randn('state', 0); % Data error noise seed
normrnd('state', 0); % Seed for multi-freq stochastic sampling

% Specify Model Type
soln = 'confined';

% Aquifer Geometry
r = 10;

% Aquifer Parameters
D_true = 2;
T_true = -8;
S_true = T_true - D_true;

% Define Parameter Surface
T_vec = (T_true-10:5e-2:T_true+10);
S_vec = (S_true-10:5e-2:S_true+10);
[T, S] = meshgrid(T_vec, S_vec);

% Pumping Amplitude (m^3/s)
Q_max = 7e-5;

% Sampling Frequency
dt = 1/8;
data_err = sqrt(1e-4); % Data error variance (Assumes 1 cm data measurement error)

% LM Inversion Inital Parameters
delta = [0.1 0.1];
lambda = 1e1;
s_init = [T_true-2; S_true-2];
% s_init = [-15; -14]; % Drives inversion to local minimum

% Stimulation Period Vector
Pv = {30, 90, 180, [30 90], [30 180], [30 90 180]};

for w = 1 : numel(Pv)
    
P = Pv{w};
omega = (2*pi) ./ P;
% Generate Test List
synth_data = [];
for i = 1 : numel(r)
    for j = 1 : numel(P)
     synth_data = [synth_data;...
                   P(j) omega(j) Q_max r(i)]; 
    end
end

num_obs = numel(synth_data(:,1));
% Generate true phasors w/o noise
y_mod = @(s) RasSoln(synth_data, s, soln);
y_true = y_mod([T_true; S_true]);

%% LM Inversion
% Generates noisy signal using Gaussian noise under i.i.d. assumption
phasor = [y_true(1:2:end-1) y_true(2:2:end)];
for i = 1 : num_obs
    t = [0 : dt : 5.*synth_data(i,1)]';
    signal = (phasor(i,1) .* cos(synth_data(i,2) .*t)) +...
             (-phasor(i,2) .* sin(synth_data(i,2) .* t));
    noise = data_err .* randn(size(t));
    sig_noise = signal + noise;
    [data_cov{i}, phasor_noise(i,:)] = periodic_LS_fit(t, sig_noise, synth_data(i,1));
    y_stddev(:,i) = 1.96 * sqrt(diag(data_cov{i}));
end
y_noise = zeros(2*num_obs,1);
y_noise(1:2:end-1) = real(phasor_noise);
y_noise(2:2:end) = imag(phasor_noise);

% Inverse data error covariance matrix
R_inv = inv(blkdiag(data_cov{1:end}));

% Conduct Inversion
% Define objective function
obj_fxn = @(s) (1/2) * (y_true - y_mod(s))' * R_inv * (y_true - y_mod(s));
[s_opt{w}, ~] = Lev_Marq(synth_data, s_init, y_noise, R_inv, lambda, delta, soln);
misfit{w} = y_true - y_mod(s_opt{w});
opt_norm{w} = obj_fxn(s_opt{w});

% Linearized uncertainty analysis
J = jacob(s_opt{w}, delta, synth_data, soln);
param_cov = inv(J' * R_inv * J);
param_sd = 1.96 * sqrt(diag(param_cov));
param_CI = [s_opt{w}-param_sd s_opt{w}+param_sd];

% Chi^2 error ellipse
[e_vec, e_val] = eig(inv(param_cov));
del = sqrt(chi2inv(0.95,2));
theta = [0 : 1e-2 : 2*pi]';

% 95% Confidence Ellipse
ell = zeros(length(theta),2);
% D component
ell(:,1) = s_opt{w}(1) +...
         ((del/sqrt(e_val(1,1))) .* e_vec(1,1) .* cos(theta)+...
          (del/sqrt(e_val(2,2))) .* e_vec(1,2) .* sin(theta));
% T component
ell(:,2) = s_opt{w}(2) +...
         ((del/sqrt(e_val(1,1))) .* e_vec(2,1) .* cos(theta)+...
          (del/sqrt(e_val(2,2))) .* e_vec(2,2) .* sin(theta));

err_ell{w} = ell;

%% Synthetic Parameter Space

% Determine model norm w.r.t. true model
mod_norm = zeros(size(T));
num_cells = numel(T);

for idx = 1 : num_cells
    mod_norm(idx) = obj_fxn([T(idx) S(idx)]);
end
mod_err{w} = mod_norm;

%% Sensitivity Analysis

% Uncertainty vs signal length
t_max = 60*60; % Total test time in seconds
[time_tot{w}, T_stddev{w}, S_stddev{w}] = SigLenUnc(synth_data, phasor, [T_true-1; S_true-1], dt, t_max, data_err, lambda, delta, soln);

% Data Error Uncertainty
data_err_sens = sqrt(2.5e-5); % Assumes 5 mm data measurement error
[time_data_sens{w}, T_unc_data_sens{w}, S_unc_data_sens{w}] = SigLenUnc(synth_data, phasor, [T_true-1; S_true-1], dt, t_max, data_err_sens, lambda, delta, soln);

% Temporal Sampling Resolution Uncertainty Sensitivity
dt_sens = 1/125;
[time_dt_sens{w}, T_unc_dt_sens{w}, S_unc_dt_sens{w}] = SigLenUnc(synth_data, phasor, [T_true-1; S_true-1], dt_sens, t_max, data_err, lambda, delta, soln);

% Radial Distance Uncertainty
r_sens = 20;
test_list = [];
for i = 1 : numel(r_sens)
    for j = 1 : numel(P)
     test_list = [test_list;...
                  P(j) omega(j) Q_max r_sens(i)]; 
    end
end

y_sens = RasSoln(test_list, [T_true; S_true], soln);
phasor_sens = [y_sens(1:2:end-1) y_sens(2:2:end)];
[time_r_sens{w}, T_unc_r_sens{w}, S_unc_r_sens{w}] = SigLenUnc(test_list, phasor_sens, [T_true-1; S_true-1], dt, t_max, data_err, lambda, delta, soln);

end

%% Figures

% Parameter Uncertainty 
col = {[0 0.4470 0.7410], [0.8500 0.3250 0.0980], [0.9290 0.6940 0.1250],...
       [0.4940 0.1840 0.5560], [0.4660 0.6740 0.1880], [0.6350 0.0780 0.1840]};
mkr = {'v', 'd', '^', 's', '>', 'o'};
leg = {'P = 30 s', 'P = 90 s', 'P = 180 s',...
       'P = 30 s & 90 s', 'P = 30 s & 180 s', 'P = 30 s, 90 s, & 180 s'};

% Figure 2
figure(2)
clf
for h = 1 : numel(Pv)
subplot(1,2,1)
ax = gca;
hold on
plot(time_tot{h}./60, T_stddev{h}, mkr{h}, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', col{h},...
    'MarkerSize', 8, 'DisplayName', leg{h})
grid on 
% xlim([time_tot{h}(1)/60 time_tot{h}(end)/60])
xlim([0 time_tot{h}(end)/60])
ylim([0 0.15])
ax.XTick = [0:10:60];
ax.YTick = [0:0.03:0.15];
xlabel('Time (min)')
ylabel('\sigma ln(T [m^2/s])')
legend show
ax.FontSize = 18;

subplot(1,2,2)
ax = gca;
hold on
plot(time_tot{h}./60, S_stddev{h}, mkr{h}, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', col{h},...
    'MarkerSize', 8, 'DisplayName', leg{h})
grid on
% xlim([time_tot{h}(1)/60 time_tot{h}(end)/60])
xlim([0 time_tot{h}(end)/60])
ylim([0 0.15])
ax.XTick = [0:10:60];
ax.YTick = [0:0.03:0.15];
xlabel('Time (min)')
ylabel('\sigma ln(S [-])')
legend show
ax.FontSize = 18;
end
set(gcf, 'Position', [0 0 1900 600])

% Figure 3
figure(3)
clf
subplot(1,2,1)
ax = gca;
plot(time_tot{end}./60, T_stddev{end}, ['-' mkr{1}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerSize', 8,...
     'Color', [0 0.4470 0.7410], 'LineWidth', 2)
hold on
plot(time_tot{end}./60, T_unc_data_sens{end}, ['-' mkr{2}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4940 0.1840 0.5560], 'MarkerSize', 8,...
     'Color', [0.4940 0.1840 0.5560], 'LineWidth', 2)
plot(time_tot{end}./60, T_unc_dt_sens{end}, ['-' mkr{3}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerSize', 8,...
     'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2)
plot(time_tot{end}./60, T_unc_r_sens{end}, ['-' mkr{4}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.9290 0.6940 0.1250], 'MarkerSize', 8,...
     'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2)
grid on
xlim([time_tot{end}(1)/60 time_tot{end}(end)/60])
ylim([0 7e-2])
xlabel('Time (min)')
ylabel('\sigma ln(T [m^2/s])')
ax.FontSize = 18;
l1 = legend('Base Case', '\sigma_{noise} = 5 mm', 'dt = 125 Hz', 'd = 20 m');
l1.FontSize = 18;

subplot(1,2,2)
ax = gca;
plot(time_tot{end}./60, S_stddev{end}, ['-' mkr{1}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerSize', 8,...
     'Color', [0 0.4470 0.7410], 'LineWidth', 2)
hold on
plot(time_tot{end}./60, S_unc_data_sens{end}, ['-' mkr{2}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4940 0.1840 0.5560], 'MarkerSize', 8,...
     'Color', [0.4940 0.1840 0.5560], 'LineWidth', 2)
plot(time_tot{end}./60, S_unc_dt_sens{end}, ['-' mkr{3}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerSize', 8,...
     'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2)
plot(time_tot{end}./60, S_unc_r_sens{end}, ['-' mkr{4}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.9290 0.6940 0.1250], 'MarkerSize', 8,...
     'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2)
grid on
xlim([time_tot{end}(1)/60 time_tot{end}(end)/60])
ylim([0 7e-2])
xlabel('Time (min)')
ylabel('\sigma ln(S [-])')
ax.FontSize = 18;
l2 = legend('Base Case', '\sigma_{noise} = 5 mm', 'dt = 125 Hz', 'd = 20 m');
l2.FontSize = 18;
set(gcf, 'Position', [0 0 1900 600])

% Figure 4
figure(4)
clf
for f = 1 : 3
subplot(2,3,f)
ax = gca;
hold on
contour(T, S, log10(mod_err{f}), 20, 'LineWidth', 2)
plot(err_ell{f}(:,1), err_ell{f}(:,2), '-', 'Color', [0.6350, 0.0780, 0.1840], 'LineWidth', 2);
axis([min(T_vec) -4 -15 -8])
ax.XTick = [min(T_vec):2:-4];
grid on
xlabel('ln(T [m^2/s])')
ylabel('ln(S [-])')
ax.FontSize = 18;
c = colorbar;
caxis([0 14])
c.Ticks = [0:2:14];
c.Label.String = 'log_{10}(Model Norm)';
c.FontSize = 18;

subplot(2,3,f+3)
ax = gca;
hold on
contour(T, S, log10(mod_err{f}), 20, 'LineWidth', 2);
p1 = plot(T_true, S_true, 'kd', 'MarkerSize', 10, 'MarkerFaceColor', [0.8500, 0.3250, 0.0980], 'LineWidth', 2);
p2 = plot(s_opt{f}(1), s_opt{f}(2), 'k^', 'LineWidth', 2,...
    'MarkerSize', 12, 'MarkerFaceColor', [0.4660, 0.6740, 0.1880]);
p3 = plot(err_ell{f}(:,1), err_ell{f}(:,2), '-', 'Color', [0.6350, 0.0780, 0.1840], 'LineWidth', 2);
axis([T_true-0.5 T_true+0.5 S_true-0.5 S_true+0.5])
grid on
xlabel('ln(T [m^2/s])')
ylabel('ln(S [-])')
ax.FontSize = 18;
legend([p1, p2, p3], 's_{true}', 's_{opt}', 's_{unc}')

end
set(gcf, 'Position', [0 0 1900 900])

% Figure 5
figure(5)
clf
for g = 1 : 3
subplot(2,3,g)
ax = gca;
hold on
contour(T, S, log10(mod_err{g+3}), 20, 'LineWidth', 2)
plot(err_ell{g+3}(:,1), err_ell{g+3}(:,2), '-', 'Color', [0.6350, 0.0780, 0.1840], 'LineWidth', 2);
axis([min(T_vec) -4 -15 -8])
ax.XTick = [min(T_vec):2:-4];
grid on
xlabel('ln(T [m^2/s])')
ylabel('ln(S [-])')
ax.FontSize = 18;
c = colorbar;
caxis([0 14])
c.Ticks = [0:2:14];
c.Label.String = 'log_{10}(Model Norm)';
c.FontSize = 18;

subplot(2,3,g+3)
ax = gca;
hold on
contour(T, S, log10(mod_err{g+3}), 20, 'LineWidth', 2);
p1 = plot(T_true, S_true, 'kd', 'MarkerSize', 10, 'MarkerFaceColor', [0.8500, 0.3250, 0.0980], 'LineWidth', 2);
p2 = plot(s_opt{g+3}(1), s_opt{g+3}(2), 'k^', 'LineWidth', 2,...
    'MarkerSize', 12, 'MarkerFaceColor', [0.4660, 0.6740, 0.1880]);
p3 = plot(err_ell{g+3}(:,1), err_ell{g+3}(:,2), '-', 'Color', [0.6350, 0.0780, 0.1840], 'LineWidth', 2);
axis([T_true-0.5 T_true+0.5 S_true-0.5 S_true+0.5])
grid on
xlabel('ln(T [m^2/s])')
ylabel('ln(S [-])')
ax.FontSize = 18;
legend([p1, p2, p3], 's_{true}', 's_{opt}', 's_{unc}')

end
set(gcf, 'Position', [0 0 1900 900])

%% Supplemental Figures
% Figure S1
for i = 1 : numel(Pv)
figure
clf   
subplot(1,2,1)
ax = gca;
plot(time_tot{i}./60, T_stddev{i}, ['-' mkr{1}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerSize', 8,...
     'Color', [0 0.4470 0.7410], 'LineWidth', 2)
hold on
plot(time_tot{i}./60, T_unc_data_sens{i}, ['-' mkr{2}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4940 0.1840 0.5560], 'MarkerSize', 8,...
     'Color', [0.4940 0.1840 0.5560], 'LineWidth', 2)
plot(time_tot{i}./60, T_unc_dt_sens{i}, ['-' mkr{3}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerSize', 8,...
     'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2)
plot(time_tot{i}./60, T_unc_r_sens{i}, ['-' mkr{4}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.9290 0.6940 0.1250], 'MarkerSize', 8,...
     'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2)
grid on
xlim([time_tot{i}(1)/60 time_tot{i}(end)/60])
if i==1
    ylim([0 max(T_unc_r_sens{i})])
else
    ylim([0 0.15])
end
xlabel('Time (min)')
ylabel('\sigma ln(T [m^2/s])')
ax.FontSize = 18;
l1 = legend('Base Case', '\sigma_{noise} = 5 mm', 'dt = 125 Hz', 'd = 20 m');
l1.FontSize = 18;

subplot(1,2,2)
ax = gca;
plot(time_tot{i}./60, S_stddev{i}, ['-' mkr{1}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerSize', 8,...
     'Color', [0 0.4470 0.7410], 'LineWidth', 2)
hold on
plot(time_tot{i}./60, S_unc_data_sens{i}, ['-' mkr{2}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4940 0.1840 0.5560], 'MarkerSize', 8,...
     'Color', [0.4940 0.1840 0.5560], 'LineWidth', 2)
plot(time_tot{i}./60, S_unc_dt_sens{i}, ['-' mkr{3}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerSize', 8,...
     'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2)
plot(time_tot{i}./60, S_unc_r_sens{i}, ['-' mkr{4}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.9290 0.6940 0.1250], 'MarkerSize', 8,...
     'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2)
grid on
xlim([time_tot{i}(1)/60 time_tot{i}(end)/60])
if i==1
    ylim([0 max(S_unc_r_sens{i})])
else
    ylim([0 0.15])
end
xlabel('Time (min)')
ylabel('\sigma ln(S [-])')
ax.FontSize = 18;
l2 = legend('Base Case', '\sigma_{noise} = 5 mm', 'dt = 125 Hz', 'd = 20 m');
l2.FontSize = 18;
set(gcf, 'Position', [0 0 1900 600])
end