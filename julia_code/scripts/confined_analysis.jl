# confined_analysis.jl
# Fully confined aquifer single and multi-frequency oscillatory flow test analysis.
# Generates parameter space, LM inversion, linearized uncertainty, and signal
# length sensitivity analysis for confined aquifer conditions.
#
# Reference:
#   Patterson, J.R. & Cardiff, M.A. (2022). Aquifer Characterization and
#   Uncertainty in Multi-Frequency Oscillatory Flow Tests: Approach and Insights.
#   Groundwater, 60(2), 180-191. https://doi.org/10.1111/gwat.13134

using Pkg
Pkg.activate(joinpath(@__DIR__, "../julia_code"))

using MofiInversion
using CairoMakie
using Random

# ── Random seed for reproducibility ──────────────────────────────────────────
Random.seed!(0)

# ── Aquifer model ─────────────────────────────────────────────────────────────
model = ConfinedAquifer()

# ── Aquifer geometry ──────────────────────────────────────────────────────────
r = 10.0                    # radial distance [m]

# ── True aquifer parameters (log-space) ───────────────────────────────────────
D_true = 2.0
T_true = -8.0
S_true = T_true - D_true

# ── Pumping parameters ────────────────────────────────────────────────────────
Q_max = 7e-5                # max pumping rate [m³/s]

# ── Sampling and noise ────────────────────────────────────────────────────────
dt       = 1/8              # sampling interval [s]
data_err = 1e-2             # observation noise std dev [m] (1 cm)

# ── Inversion parameters ──────────────────────────────────────────────────────
δ      = [0.1, 0.1]         # finite difference perturbations
λ      = 1e1                # initial LM stabilization parameter
s_init = [T_true - 2.0, S_true - 2.0]

# ── Parameter grid ────────────────────────────────────────────────────────────
T_vec = collect(T_true - 10 : 5e-2 : T_true + 10)
S_vec = collect(S_true - 10 : 5e-2 : S_true + 10)

# ── Stimulation period sets ───────────────────────────────────────────────────
Pv = [
    [30.0],
    [90.0],
    [180.0],
    [30.0, 90.0],
    [30.0, 180.0],
    [30.0, 90.0, 180.0]
]

# ── Storage arrays ────────────────────────────────────────────────────────────
num_cases    = length(Pv)
s_opt        = Vector{Vector{Float64}}(undef, num_cases)
err_ell      = Vector{Matrix{Float64}}(undef, num_cases)
mod_err      = Vector{Matrix{Float64}}(undef, num_cases)
time_tot     = Vector{Vector{Float64}}(undef, num_cases)
T_stddev     = Vector{Vector{Float64}}(undef, num_cases)
S_stddev     = Vector{Vector{Float64}}(undef, num_cases)
time_data_sens    = Vector{Vector{Float64}}(undef, num_cases)
T_unc_data_sens   = Vector{Vector{Float64}}(undef, num_cases)
S_unc_data_sens   = Vector{Vector{Float64}}(undef, num_cases)
time_dt_sens      = Vector{Vector{Float64}}(undef, num_cases)
T_unc_dt_sens     = Vector{Vector{Float64}}(undef, num_cases)
S_unc_dt_sens     = Vector{Vector{Float64}}(undef, num_cases)
time_r_sens       = Vector{Vector{Float64}}(undef, num_cases)
T_unc_r_sens      = Vector{Vector{Float64}}(undef, num_cases)
S_unc_r_sens      = Vector{Vector{Float64}}(undef, num_cases)

# ── Main analysis loop ────────────────────────────────────────────────────────
for w in 1:num_cases

    P     = Pv[w]
    ω     = 2π ./ P
    num_P = length(P)

    # Build test list — columns: [Period, ω, Q_max, r]
    test_list = Matrix{Float64}(undef, num_P, 4)
    for j in 1:num_P
        test_list[j, :] = [P[j], ω[j], Q_max, r]
    end
    num_obs = size(test_list, 1)

    # Define forward model and objective function closures
    fwd_func = s -> rasmussen_phasor(test_list, s, model)
    y_true   = fwd_func([T_true, S_true])
    obj_func = s -> 0.5 * dot(y_true - fwd_func(s), inv(diagm(ones(length(y_true)))) * (y_true - fwd_func(s)))

    # ── Generate noisy signals and extract phasors ────────────────────────────
    data_covs = Vector{Matrix{Float64}}(undef, num_obs)
    phasors   = Vector{ComplexF64}(undef, num_obs)

    for i in 1:num_obs
        t         = collect(0.0 : dt : 5.0 * test_list[i, 1])
        signal    = (y_true[i] .* cos.(test_list[i, 2] .* t)) .+
                   (-y_true[num_obs+i] .* sin.(test_list[i, 2] .* t))
        sig_noise = signal .+ data_err .* randn(length(t))
        fit       = periodic_ls_fit(t, sig_noise, [test_list[i, 1]])
        data_covs[i] = fit.data_cov
        phasors[i]   = fit.phasors[1]
    end

    # Build data vector and inverse covariance matrix
    y_noise                          = zeros(Float64, 2 * num_obs)
    y_noise[1:num_obs]               = real.(phasors)
    y_noise[num_obs+1:2*num_obs]     = imag.(phasors)
    R_inv                            = inv(blockdiag(data_covs...))

    # ── LM inversion ─────────────────────────────────────────────────────────
    inv_result  = grad_inv_lm(y_noise, s_init, fwd_func, obj_func, R_inv, λ, δ)
    s_opt[w]    = inv_result.s_hat
    s_step      = inv_result.s_history

    println("w=$w  lnT=$(round(s_opt[w][1], digits=3))  lnS=$(round(s_opt[w][2], digits=3))  converged=$(inv_result.converged)")

    # ── Linearized uncertainty analysis ───────────────────────────────────────
    unc          = param_uncertainty(s_opt[w], δ, fwd_func, R_inv)
    region       = confidence_region(unc.param_cov, s_opt[w], model)
    err_ell[w]   = region.points

    # ── Parameter space grid search ───────────────────────────────────────────
    mod_err[w] = parameter_space_search(obj_func, T_vec, S_vec)

    # ── Signal length uncertainty — base case ─────────────────────────────────
    t_max      = 60.0 * 60.0              # 1 hour [s]
    phasor_mat = [real.(phasors) imag.(phasors)]

    sig_unc  = sig_len_unc(test_list, phasor_mat, [T_true - 1.0, S_true - 1.0],
                            dt, t_max, data_err, λ, δ, fwd_func, obj_func, model)
    time_tot[w]  = sig_unc.t_save
    T_stddev[w]  = sig_unc.param_stddev[:, 1]
    S_stddev[w]  = sig_unc.param_stddev[:, 2]

    # ── Signal length uncertainty — data error sensitivity ────────────────────
    data_err_sens = 2.5e-5              # 5 mm noise
    sig_unc_data  = sig_len_unc(test_list, phasor_mat, [T_true - 1.0, S_true - 1.0],
                                 dt, t_max, data_err_sens, λ, δ, fwd_func, obj_func, model)
    time_data_sens[w]  = sig_unc_data.t_save
    T_unc_data_sens[w] = sig_unc_data.param_stddev[:, 1]
    S_unc_data_sens[w] = sig_unc_data.param_stddev[:, 2]

    # ── Signal length uncertainty — temporal sampling sensitivity ─────────────
    dt_sens      = 1/125
    sig_unc_dt   = sig_len_unc(test_list, phasor_mat, [T_true - 1.0, S_true - 1.0],
                                dt_sens, t_max, data_err, λ, δ, fwd_func, obj_func, model)
    time_dt_sens[w]  = sig_unc_dt.t_save
    T_unc_dt_sens[w] = sig_unc_dt.param_stddev[:, 1]
    S_unc_dt_sens[w] = sig_unc_dt.param_stddev[:, 2]

    # ── Signal length uncertainty — radial distance sensitivity ───────────────
    r_sens     = 20.0
    test_list_r = Matrix{Float64}(undef, num_P, 4)
    for j in 1:num_P
        test_list_r[j, :] = [P[j], ω[j], Q_max, r_sens]
    end

    fwd_func_r   = s -> rasmussen_phasor(test_list_r, s, model)
    y_sens       = fwd_func_r([T_true, S_true])
    phasor_sens  = [y_sens[1:num_P]  y_sens[num_P+1:2*num_P]]

    sig_unc_r    = sig_len_unc(test_list_r, phasor_sens, [T_true - 1.0, S_true - 1.0],
                                dt, t_max, data_err, λ, δ, fwd_func_r, obj_func_r, model)
    time_r_sens[w]  = sig_unc_r.t_save
    T_unc_r_sens[w] = sig_unc_r.param_stddev[:, 1]
    S_unc_r_sens[w] = sig_unc_r.param_stddev[:, 2]

end # end main loop

# ── Plot styling ──────────────────────────────────────────────────────────────
colors = [
    RGBf(0, 0.4470, 0.7410),
    RGBf(0.8500, 0.3250, 0.0980),
    RGBf(0.9290, 0.6940, 0.1250),
    RGBf(0.4940, 0.1840, 0.5560),
    RGBf(0.4660, 0.6740, 0.1880),
    RGBf(0.6350, 0.0780, 0.1840)
]
markers = [:dtriangle, :diamond, :utriangle, :rect, :rtriangle, :circle]
labels  = ["P = 30 s", "P = 90 s", "P = 180 s",
           "P = 30 s & 90 s", "P = 30 s & 180 s", "P = 30 s, 90 s, & 180 s"]

# ── Gradient path figure ──────────────────────────────────────────────────────
# Uses last w iteration — rerun inversion for last case to recover s_step
let
    w         = num_cases
    P         = Pv[w]
    ω         = 2π ./ P
    num_P     = length(P)
    test_list = hcat([[P[j], ω[j], Q_max, r] for j in 1:num_P]...)'
    fwd_func  = s -> rasmussen_phasor(test_list, s, model)
    y_true    = fwd_func([T_true, S_true])
    obj_func  = s -> 0.5 * dot(y_true - fwd_func(s), I * (y_true - fwd_func(s)))
    inv_result = grad_inv_lm(y_true, s_init, fwd_func, obj_func, I(length(y_true)), λ, δ)
    s_step     = inv_result.s_history

    fig = Figure(resolution = (800, 600))
    ax  = Axis(fig[1, 1],
               xlabel = "ln(T [m²/s])", ylabel = "ln(S [-])",
               titlesize = 18, xlabelsize = 18, ylabelsize = 18)
    contour!(ax, T_vec, S_vec, log10.(mod_err[w])', levels = 20, linewidth = 2)
    lines!(ax, s_step[:, 1], s_step[:, 2],
           color = :black, linewidth = 2,
           label = "Gradient Step")
    scatter!(ax, s_step[:, 1], s_step[:, 2],
             marker = :utriangle, markersize = 10,
             color = colors[2], strokecolor = :black, strokewidth = 1)
    xlims!(ax, -15, -4)
    ylims!(ax, -15, -8)
    axislegend(ax, fontsize = 18)
    Colorbar(fig[1, 2], limits = (0, 14), label = "log₁₀(Data Misfit)",
             labelsize = 18, ticksize = 18)
    display(fig)
end

# ── Parameter space + confidence ellipse (last case, zoomed) ─────────────────
let
    w   = num_cases
    fig = Figure(resolution = (800, 600))
    ax  = Axis(fig[1, 1],
               xlabel = "ln(T [m²/s])", ylabel = "ln(S [-])",
               xlabelsize = 18, ylabelsize = 18)
    contour!(ax, T_vec, S_vec, log10.(mod_err[w])', levels = 20, linewidth = 2)
    scatter!(ax, [T_true], [S_true],
             marker = :diamond, markersize = 10,
             color = colors[5], strokecolor = :black, strokewidth = 2,
             label = "params_true")
    scatter!(ax, [s_opt[w][1]], [s_opt[w][2]],
             marker = :utriangle, markersize = 12,
             color = colors[2], strokecolor = :black, strokewidth = 2,
             label = "params_opt")
    lines!(ax, err_ell[w][:, 1], err_ell[w][:, 2],
           color = colors[6], linewidth = 2, linestyle = :dash,
           label = "params_unc")
    xlims!(ax, T_true - 0.5, T_true + 0.5)
    ylims!(ax, S_true - 0.5, S_true + 0.5)
    axislegend(ax, fontsize = 18)
    display(fig)
end

# ── Figure 2 — uncertainty vs signal length, all period sets ─────────────────
let
    fig = Figure(resolution = (1900, 600))
    ax1 = Axis(fig[1, 1], xlabel = "Time (min)", ylabel = "σ ln(T [m²/s])",
               xlabelsize = 18, ylabelsize = 18)
    ax2 = Axis(fig[1, 2], xlabel = "Time (min)", ylabel = "σ ln(S [-])",
               xlabelsize = 18, ylabelsize = 18)

    for h in 1:num_cases
        t_min = time_tot[h] ./ 60
        scatter!(ax1, t_min, T_stddev[h],
                 marker = markers[h], markersize = 8,
                 color = colors[h], strokecolor = :black, strokewidth = 1,
                 label = labels[h])
        scatter!(ax2, t_min, S_stddev[h],
                 marker = markers[h], markersize = 8,
                 color = colors[h], strokecolor = :black, strokewidth = 1,
                 label = labels[h])
    end

    for ax in [ax1, ax2]
        xlims!(ax, 0, maximum(time_tot[end]) / 60)
        ylims!(ax, 0, 0.15)
        ax.xticks = 0:10:60
        ax.yticks = 0:0.03:0.15
        axislegend(ax, fontsize = 14)
    end
    display(fig)
end

# ── Figure 3 — sensitivity comparison, multi-freq case ───────────────────────
let
    w   = num_cases
    t_m = time_tot[w] ./ 60

    sens_colors  = [colors[1], colors[4], colors[5], colors[3]]
    sens_markers = markers[1:4]
    sens_labels  = ["Base Case", "σ_noise = 5 mm", "dt = 125 Hz", "d = 20 m"]

    T_sens_data = [T_stddev[w], T_unc_data_sens[w], T_unc_dt_sens[w], T_unc_r_sens[w]]
    S_sens_data = [S_stddev[w], S_unc_data_sens[w], S_unc_dt_sens[w], S_unc_r_sens[w]]

    fig = Figure(resolution = (1900, 600))
    ax1 = Axis(fig[1, 1], xlabel = "Time (min)", ylabel = "σ ln(T [m²/s])",
               xlabelsize = 18, ylabelsize = 18)
    ax2 = Axis(fig[1, 2], xlabel = "Time (min)", ylabel = "σ ln(S [-])",
               xlabelsize = 18, ylabelsize = 18)

    for (k, (Td, Sd)) in enumerate(zip(T_sens_data, S_sens_data))
        lines!(ax1, t_m, Td, color = sens_colors[k], linewidth = 2)
        scatter!(ax1, t_m, Td, marker = sens_markers[k], markersize = 8,
                 color = sens_colors[k], strokecolor = :black, strokewidth = 1,
                 label = sens_labels[k])
        lines!(ax2, t_m, Sd, color = sens_colors[k], linewidth = 2)
        scatter!(ax2, t_m, Sd, marker = sens_markers[k], markersize = 8,
                 color = sens_colors[k], strokecolor = :black, strokewidth = 1,
                 label = sens_labels[k])
    end

    for ax in [ax1, ax2]
        xlims!(ax, t_m[1], t_m[end])
        ylims!(ax, 0, 7e-2)
        axislegend(ax, fontsize = 18)
    end
    display(fig)
end

# ── Figures 4 & 5 — parameter space + ellipse, single and multi-freq ─────────
for (fig_cases, fig_num) in [((1:3), 4), ((4:6), 5)]
    fig = Figure(resolution = (1900, 900))
    for (col_idx, f) in enumerate(fig_cases)
        # Full parameter space
        ax_top = Axis(fig[1, col_idx],
                      xlabel = "ln(T [m²/s])", ylabel = "ln(S [-])",
                      xlabelsize = 18, ylabelsize = 18)
        contour!(ax_top, T_vec, S_vec, log10.(mod_err[f])', levels = 20, linewidth = 2)
        lines!(ax_top, err_ell[f][:, 1], err_ell[f][:, 2],
               color = colors[6], linewidth = 2)
        xlims!(ax_top, minimum(T_vec), -4)
        ylims!(ax_top, -15, -8)
        Colorbar(fig[1, col_idx + length(fig_cases)],
                 limits = (0, 14), label = "log₁₀(Model Norm)",
                 labelsize = 18)

        # Zoomed view
        ax_bot = Axis(fig[2, col_idx],
                      xlabel = "ln(T [m²/s])", ylabel = "ln(S [-])",
                      xlabelsize = 18, ylabelsize = 18)
        contour!(ax_bot, T_vec, S_vec, log10.(mod_err[f])', levels = 20, linewidth = 2)
        scatter!(ax_bot, [T_true], [S_true],
                 marker = :diamond, markersize = 10,
                 color = colors[2], strokecolor = :black, strokewidth = 2,
                 label = "s_true")
        scatter!(ax_bot, [s_opt[f][1]], [s_opt[f][2]],
                 marker = :utriangle, markersize = 12,
                 color = colors[5], strokecolor = :black, strokewidth = 2,
                 label = "s_opt")
        lines!(ax_bot, err_ell[f][:, 1], err_ell[f][:, 2],
               color = colors[6], linewidth = 2, label = "s_unc")
        xlims!(ax_bot, T_true - 0.5, T_true + 0.5)
        ylims!(ax_bot, S_true - 0.5, S_true + 0.5)
        axislegend(ax_bot, fontsize = 14)
    end
    display(fig)
end

# ── Supplemental Figure S1 — per-case sensitivity ────────────────────────────
sens_colors  = [colors[1], colors[4], colors[5], colors[3]]
sens_markers = markers[1:4]
sens_labels  = ["Base Case", "σ_noise = 5 mm", "dt = 125 Hz", "d = 20 m"]

for i in 1:num_cases
    t_m = time_tot[i] ./ 60

    T_sens = [T_stddev[i], T_unc_data_sens[i], T_unc_dt_sens[i], T_unc_r_sens[i]]
    S_sens = [S_stddev[i], S_unc_data_sens[i], S_unc_dt_sens[i], S_unc_r_sens[i]]

    T_ylim = i == 1 ? maximum(T_unc_r_sens[i]) : 0.15
    S_ylim = i == 1 ? maximum(S_unc_r_sens[i]) : 0.15

    fig = Figure(resolution = (1900, 600))
    ax1 = Axis(fig[1, 1], xlabel = "Time (min)", ylabel = "σ ln(T [m²/s])",
               xlabelsize = 18, ylabelsize = 18)
    ax2 = Axis(fig[1, 2], xlabel = "Time (min)", ylabel = "σ ln(S [-])",
               xlabelsize = 18, ylabelsize = 18)

    for (k, (Td, Sd)) in enumerate(zip(T_sens, S_sens))
        lines!(ax1, t_m, Td, color = sens_colors[k], linewidth = 2)
        scatter!(ax1, t_m, Td, marker = sens_markers[k], markersize = 8,
                 color = sens_colors[k], strokecolor = :black, strokewidth = 1,
                 label = sens_labels[k])
        lines!(ax2, t_m, Sd, color = sens_colors[k], linewidth = 2)
        scatter!(ax2, t_m, Sd, marker = sens_markers[k], markersize = 8,
                 color = sens_colors[k], strokecolor = :black, strokewidth = 1,
                 label = sens_labels[k])
    end

    for (ax, ylim) in [(ax1, T_ylim), (ax2, S_ylim)]
        xlims!(ax, t_m[1], t_m[end])
        ylims!(ax, 0, ylim)
        axislegend(ax, fontsize = 18)
    end
    display(fig)
end