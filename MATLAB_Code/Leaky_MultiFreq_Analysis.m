% Multi-Frequency Analysis Leaky Conceptual Model (Patterson and Cardiff, 2022)
% Code developed by Jeremy Patterson
% Created Dec 2020; Updated May 2021

% This code generates the multi-frequency analysis seen in the section "Leaky Aquifer System" (Patterson and Cardiff, 2022)
% Patterson, J. R., & Cardiff, M. (2022). Aquifer Characterization and Uncertainty in Multi-Frequency Oscillatory Flow Tests: Approach and Insights. Groundwater, 60(2),
% 180–191. https://doi.org/10.1111/gwat.13134


%% Clean Environment
close all; clear; clc

%% Specify Directory
% Specify the directory location of the folder Func_Lib, which contains the needed function files to execute this code
addpath('/.../.../') 

%% Create Synthetic Data
% Seed the random number generators
randn('state', 0); % Data noise
normrnd('state', 0); % Seed for multi-freq stochastic sampling

% Specify model type
soln = 'leaky';

% Aquifer Geometry
r = 10; % Radial distance (m)

% True Aquifer Parameters (in ln space)
D_true = 2;   %ln(Diffusivity (m^2/s))
T_true = -8;  %ln(Transmissivity (m^2/s))
S_true = T_true-D_true;
L_true = -21; %ln(Leakance (s^-1))

% Pumping Parameters
Q_max = 7e-5; % Max pumping rate (m^3/s)

% Stimulation Period Vector (s)
Pv = {[3600 5400], [3600 7200], [3600 5400 7200]};

dt = 1/8; % Sampling Frequency
data_err = 1e-4; % Observation signal noise variance

% LM Inversion Initial Parameters
s_init = [T_true-2; S_true-2; L_true+2]; % Parameter initial guess
delta = [0.1; 0.1; 0.1]; % Parameter perturbation to caluclate Jacobian
lambda = 1e1; % LM Stabilization parameter

for w = 1 : numel(Pv)
    P = Pv{w};
    omega = (2*pi) ./ P; % Angular frequency (rad/s)
    
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
    y_true = y_mod([T_true; S_true; L_true]); %True Fourier coefficients (w/o noise)

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
    % Fourier coefficients with added noise. Odd elements are real
    % components, even elements are imaginary components.
    y_noise(1:2:end-1) = real(phasor_noise);
    y_noise(2:2:end) = imag(phasor_noise);
    
    % Inverse data error covariance matrix
    R_inv = inv(blkdiag(data_cov{1:end}));
    
    s_opt{w} = Lev_Marq(synth_data, s_init, y_noise, R_inv, lambda, delta, soln);
    
    % Linearized uncertainty analysis
    J = jacob(s_opt{w}, delta, synth_data, soln);
    param_cov = inv(J' * R_inv * J);
    param_sd = 1.96 * sqrt(diag(param_cov));
    param_CI = [s_opt{w} - param_sd; s_opt{w} + param_sd];
    
    % Chi^2 error ellipsoid
    [e_vec, e_val] = eig(inv(param_cov));
    del = sqrt(chi2inv(0.95,3));
    r_ell = del ./ sqrt(diag(e_val));
    e_vec_scaled = [e_vec(:,1)*r_ell(1) e_vec(:,2)*r_ell(2) e_vec(:,3)*r_ell(3)];
    e_vec_save{w} = e_vec_scaled;
    
    % Unit sphere
    n = 100;
    [t, s, l] = sphere(n);
    
    % Rotated / scaled parameter uncertainty ellipsoid
    coord_rot = e_vec_scaled * [t(:) s(:) l(:)]';
    t_rot{w} = reshape(coord_rot(1,:),[n n]+1) + s_opt{w}(1);
    s_rot{w} = reshape(coord_rot(2,:),[n n]+1) + s_opt{w}(2);
    l_rot{w} = reshape(coord_rot(3,:),[n n]+1) + s_opt{w}(3);
    
    %% Sensitivity Analysis
    
    % Signal Length Uncertainty
    t_max = 24*60*60; % Total test time in seconds
    [time_tot{w}, T_stddev{w}, S_stddev{w}, L_stddev{w}] = SigLenUnc(synth_data, phasor, [T_true-1; S_true-1; L_true+1], dt, t_max, data_err, lambda, delta, soln);
    
    % Observation Signal Noise Sensitivity
    data_err_sens = 2.5e-5; % Assumes 5 mm data measurement error
    [time_data_sens{w}, T_unc_data_sens{w}, S_unc_data_sens{w}, L_unc_data_sens{w}] = SigLenUnc(synth_data, phasor, [T_true-1; S_true-1; L_true+1], dt, t_max, data_err_sens, lambda, delta, soln);
    
    % Temporal Sampling Resolution Sensitivity
    dt_sens = 1/125;
    [time_dt_sens{w}, T_unc_dt_sens{w}, S_unc_dt_sens{w}, L_unc_dt_sens{w}] = SigLenUnc(synth_data, phasor, [T_true-1; S_true-1; L_true+1], dt_sens, t_max, data_err, lambda, delta, soln);
    
    % Inter-well Spacing Uncertainty
    r_sens = 20;
    test_list = [];
    for i = 1 : numel(r_sens)
        for j = 1 : numel(P)
            test_list = [test_list; ...
                P(j) omega(j) Q_max r_sens(i)];
        end
    end
    
    y_sens = RasSoln(test_list, [T_true; S_true; L_true], soln);
    phasor_sens = [y_sens(1:2:end-1) y_sens(2:2:end)];
    [time_r_sens{w}, T_unc_r_sens{w}, S_unc_r_sens{w}, L_unc_r_sens{w}] = SigLenUnc(test_list, phasor_sens, [T_true-1; S_true-1; L_true+1], dt, t_max, data_err, lambda, delta, soln);
end

%% Figures
col = {[0 0.4470 0.7410], [0.8500 0.3250 0.0980], [0.9290 0.6940 0.1250],...
       [0.4940 0.1840 0.5560], [0.4660 0.6740 0.1880], [0.6350 0.0780 0.1840]};
mkr = {'v', 'd', '^', 's', '>', 'o'};
leg = {'P = 30 s', 'P = 90 s', 'P = 180 s',...
       'P = 3600 s & 5400 s', 'P = 3600 s & 7200 s', 'P = 3600 s, 5400 s, & 7200 s'};
idx = 3;
% Figure 7
figure(7)
clf
for k = 1 : 2
    subplot(1,2,k)
    ax = gca;
    % True and Optimal Parameters
    p1 = plot3(T_true, S_true, L_true, 'kd', 'LineWidth', 2, ...
        'MarkerFaceColor', [0.8500, 0.3250, 0.0980], 'MarkerSize', 10);
    hold on
    p2 = plot3(s_opt{k}(1), s_opt{k}(2), s_opt{k}(3), 'k^', 'LineWidth', 2,...
        'MarkerFaceColor', [0.4660, 0.6740, 0.1880], 'MarkerSize', 10);
    % Error Ellipse Axes
    colormap(gray)
    p3 = surfl(t_rot{k}, s_rot{k}, l_rot{k}, [80, 90], [0.6 0.4 0.8 15]);
    p3.EdgeColor = 'none';
    p3.FaceColor = 'flat';
    p3.FaceAlpha = 0.5;
    plot3([s_opt{k}(1) s_opt{k}(1) + e_vec_save{k}(1,1)], [s_opt{k}(2) s_opt{k}(2) + e_vec_save{k}(2,1)], [s_opt{k}(3) s_opt{k}(3) + e_vec_save{k}(3,1)],...
        '-', 'Color', 'w', 'LineWidth', 2)
    plot3([s_opt{k}(1) s_opt{k}(1) + e_vec_save{k}(1,2)], [s_opt{k}(2) s_opt{k}(2) + e_vec_save{k}(2,2)], [s_opt{k}(3) s_opt{k}(3) + e_vec_save{k}(3,2)],...
        '-', 'Color', 'w', 'LineWidth', 2)
    plot3([s_opt{k}(1) s_opt{k}(1) + e_vec_save{k}(1,3)], [s_opt{k}(2) s_opt{k}(2) + e_vec_save{k}(2,3)], [s_opt{k}(3) s_opt{k}(3) + e_vec_save{k}(3,3)],...
        '-', 'Color', 'w', 'LineWidth', 2)
    surf(-8.2*ones(n+1,n+1), s_rot{k}, l_rot{k}, 'LineStyle', 'none', 'FaceAlpha', 0.2)
    surf(t_rot{k}, -9.8*ones(n+1, n+1), l_rot{k}, 'LineStyle', 'none', 'FaceAlpha', 0.2)
    surf(t_rot{k}, s_rot{k}, -23*ones(n+1, n+1), 'LineStyle', 'none', 'FaceAlpha', 0.2)
    grid on
    view([60 55])
%     axis equal
    l1 = legend([p1, p2], 's_{true}', 's_{opt}', 'Location', 'northoutside');
    l1.FontSize = 16;
    axis([T_true-0.2 T_true+0.2 S_true-0.2 S_true+0.2 L_true-2 L_true+2])
    xlabel('ln(T [m^2/s])')
    ylabel('ln(S [-])')
    zlabel('ln(L [s^{-1}])')
    ax.FontSize = 18;
end
set(gcf, 'Position', [100 100 1900 700])

% Figure 8
figure(8)
clf
for h = 1 : 3
subplot(1,3,1)
ax = gca;
hold on
plot(time_tot{h}./60, T_stddev{h}, mkr{h+3}, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', col{h+3},...
     'MarkerSize', 8, 'DisplayName', leg{h+3})
grid on 
xlim([time_tot{h}(1)/60 time_tot{3}(end)/60])
ylim([0 4e-2])
xlabel('Time (min)')
ylabel('\sigma ln(T [m^2/s])')
legend show
ax.FontSize = 18;

subplot(1,3,2)
ax = gca;
hold on
plot(time_tot{h}./60, S_stddev{h}, mkr{h+3}, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', col{h+3},...
     'MarkerSize', 8, 'DisplayName', leg{h+3})
grid on
xlim([time_tot{h}(1)/60 time_tot{3}(end)/60])
ylim([0 4e-2])
xlabel('Time (min)')
ylabel('\sigma ln(S [-])')
legend show
ax.FontSize = 18;

subplot(1,3,3)
ax = gca;
hold on
plot(time_tot{h}./60, L_stddev{h}, mkr{h+3}, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', col{h+3},...
     'MarkerSize', 8, 'DisplayName', leg{h+3})
grid on
xlim([time_tot{h}(1)/60 time_tot{3}(end)/60])
ylim([0 1.2])
xlabel('Time (min)')
ylabel('\sigma ln(L [s^{-1}])')
legend show
ax.FontSize = 18;
end
set(gcf, 'Position', [0 0 1900 500])

% Figure 9
figure(9)
clf
subplot(1,3,1)
ax = gca;
plot(time_tot{3}./60, T_stddev{3}, ['-' mkr{1}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerSize', 8,...
     'Color', [0 0.4470 0.7410], 'LineWidth', 2)
hold on
plot(time_tot{3}./60, T_unc_data_sens{3}, ['-' mkr{2}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4940 0.1840 0.5560], 'MarkerSize', 8,...
     'Color', [0.4940 0.1840 0.5560], 'LineWidth', 2)
plot(time_tot{3}./60, T_unc_dt_sens{3}, ['-' mkr{3}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerSize', 8,...
     'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2)
plot(time_tot{3}./60, T_unc_r_sens{3}, ['-' mkr{4}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.9290 0.6940 0.1250], 'MarkerSize', 8,...
     'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2)
grid on
xlim([time_tot{3}(1)/60 time_tot{3}(end)/60])
ylim([0 4e-2])
xlabel('Time (min)')
ylabel('\sigma ln(T [m^2/s])')
ax.FontSize = 18;
l1 = legend('Base Case', '\sigma_{noise} = 5 mm', 'dt = 125 Hz', 'd = 20 m');
l1.FontSize = 18;

subplot(1,3,2)
ax = gca;
plot(time_tot{3}./60, S_stddev{3}, ['-' mkr{1}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerSize', 8,...
     'Color', [0 0.4470 0.7410], 'LineWidth', 2)
hold on
plot(time_tot{3}./60, S_unc_data_sens{3}, ['-' mkr{2}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4940 0.1840 0.5560], 'MarkerSize', 8,...
     'Color', [0.4940 0.1840 0.5560], 'LineWidth', 2)
plot(time_tot{3}./60, S_unc_dt_sens{3}, ['-' mkr{3}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerSize', 8,...
     'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2)
plot(time_tot{3}./60, S_unc_r_sens{3}, ['-' mkr{4}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.9290 0.6940 0.1250], 'MarkerSize', 8,...
     'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2)
grid on
xlim([time_tot{3}(1)/60 time_tot{3}(end)/60])
ylim([0 4e-2])
xlabel('Time (min)')
ylabel('\sigma ln(S [-])')
ax.FontSize = 18;
l2 = legend('Base Case', '\sigma_{noise} = 5 mm', 'dt = 125 Hz', 'd = 20 m');
l2.FontSize = 18;

subplot(1,3,3)
ax = gca;
plot(time_tot{3}./60, L_stddev{3}, ['-' mkr{1}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerSize', 8,...
     'Color', [0 0.4470 0.7410], 'LineWidth', 2)
hold on
plot(time_tot{3}./60, L_unc_data_sens{3}, ['-' mkr{2}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4940 0.1840 0.5560], 'MarkerSize', 8,...
     'Color', [0.4940 0.1840 0.5560], 'LineWidth', 2)
plot(time_tot{3}./60, L_unc_dt_sens{3}, ['-' mkr{3}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerSize', 8,...
     'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2)
plot(time_tot{3}./60, L_unc_r_sens{3}, ['-' mkr{4}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.9290 0.6940 0.1250], 'MarkerSize', 8,...
     'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2)
grid on
xlim([time_tot{3}(1)/60 time_tot{3}(end)/60])
ylim([0 1.2])
xlabel('Time (min)')
ylabel('\sigma ln(L [s^{-1}])')
ax.FontSize = 18;
l3 = legend('Base Case', '\sigma_{noise} = 5 mm', 'dt = 125 Hz', 'd = 20 m');
l3.FontSize = 18;
set(gcf, 'Position', [0 0 1900 500])

%% Supplemental Figures

% Figure S2
for i = 1 : numel(Pv)
figure
clf   
subplot(1,3,1)
ax = gca;
plot(time_tot{i}./60, T_stddev{i}, ['-' mkr{1}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerSize', 8,...
     'Color', [0 0.4470 0.7410], 'LineWidth', 2)
hold on
plot(time_tot{i}./60, T_unc_data_sens{i}, ['-' mkr{2}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4940 0.1840 0.5560], 'MarkerSize', 8,...
     'Color', [0.4940 0.1840 0.5560], 'LineWidth', 2)
plot(time_tot{i}./60, T_unc_dt_sens{i},['-' mkr{3}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerSize', 8,...
     'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2)
plot(time_tot{i}./60, T_unc_r_sens{i}, ['-' mkr{4}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.9290 0.6940 0.1250], 'MarkerSize', 8,...
    'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2)
grid on
xlim([time_tot{i}(1)/60 time_tot{i}(end)/60])
ylim([0 5e-2])
xlabel('Time (min)')
ylabel('\sigma ln(T [m^2/s])')
ax.FontSize = 18;
l1 = legend('Base Case', '\sigma_{noise} = 5 mm', 'dt = 125 Hz', 'd = 20 m');
l1.FontSize = 18;

subplot(1,3,2)
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
ylim([0 5e-2])
xlabel('Time (min)')
ylabel('\sigma ln(S [-])')
ax.FontSize = 18;

subplot(1,3,3)
ax = gca;
plot(time_tot{i}./60, L_stddev{i}, ['-' mkr{1}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerSize', 8,...
     'Color', [0 0.4470 0.7410], 'LineWidth', 2)
hold on
plot(time_tot{i}./60, L_unc_data_sens{i}, ['-' mkr{2}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4940 0.1840 0.5560], 'MarkerSize', 8,...
     'Color', [0.4940 0.1840 0.5560], 'LineWidth', 2)
plot(time_tot{i}./60, L_unc_dt_sens{i}, ['-' mkr{3}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerSize', 8,...
     'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2)
plot(time_tot{i}./60, L_unc_r_sens{i}, ['-' mkr{4}], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.9290 0.6940 0.1250], 'MarkerSize', 8,...
     'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2)
grid on
xlim([time_tot{i}(1)/60 time_tot{i}(end)/60])
ylim([0 4])
xlabel('Time (min)')
ylabel('\sigma ln(L [s^{-1}])')
ax.FontSize = 18;
set(gcf, 'Position', [0 0 1900 500])
end