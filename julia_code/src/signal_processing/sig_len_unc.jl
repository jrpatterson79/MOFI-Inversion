# sig_len_unc.jl
# Signal length uncertainty analysis for oscillatory flow tests.
# Evaluates how parameter uncertainty evolves as a function of total
# test duration by incrementally increasing signal length and repeating
# inversion and linearized uncertainty analysis at each step.

"""
    sig_len_unc(test_list, y, s_init, dt, t_max, data_err, λ, δ,
                fwd_func, obj_func, model) -> SigLenUncResult

Evaluate parameter uncertainty as a function of total oscillatory test duration.

Incrementally increases signal length from 2 periods upward, generating noisy
synthetic signals, extracting phasor coefficients via least squares, running
LM inversion, and computing linearized uncertainty at each step until `t_max`
is reached.

# Arguments
- `test_list`:  (num_obs × 4) matrix [Period(s), ω(rad/s), Q_max(m³/s), r(m)]
- `y`:          (num_obs × 2) true phasor coefficients [real imag] (no noise)
- `s_init`:     (num_params,) initial parameter guess for inversion (log-space)
- `dt`:         sampling interval [s]
- `t_max`:      maximum total test duration [s]
- `data_err`:   observation noise standard deviation [m]
- `λ`:          initial LM stabilization parameter
- `δ`:          (num_params,) finite difference perturbations for Jacobian
- `fwd_func`:   forward model closure with signature f(s) -> Vector{Float64}
- `obj_func`:   objective function closure with signature f(s) -> Float64
- `model`:      `ConfinedAquifer()` or `LeakyAquifer()`

# Returns
`SigLenUncResult` with fields:
- `t_save`:      (num_iter,) cumulative test time at each signal length [s]
- `param_stddev`: (num_iter x num_params) 95% parameter standard deviations
                   columns ordered as [T, S] for confined, [T, S, L] for leaky
- `s_hat`:       (num_iter x num_params) optimal parameters at each signal length
"""

struct SigLenUncResult
    t_save       :: Vector{Float64}
    param_stddev :: Matrix{Float64}
    s_hat        :: Matrix{Float64}
end

function sig_len_unc(
    test_list :: Matrix{Float64},
    y         :: Matrix{Float64},
    s_init    :: Vector{Float64},
    dt        :: Float64,
    t_max     :: Float64,
    data_err  :: Float64,
    λ         :: Float64,
    δ         :: Vector{Float64},
    fwd_func  :: Function,
    obj_func  :: Function,
    model     :: AquiferModel
)

    num_obs       = size(test_list, 1)
    num_params    = length(s_init)
    signal_length = 2               # start at 2 periods
    t_total       = 0.0

    # Pre-allocate result accumulators
    t_save           = Float64[]
    param_stddev_acc = Vector{Vector{Float64}}()
    s_hat_acc        = Vector{Vector{Float64}}()

    while t_total < t_max
        t_curr    = 0.0
        data_covs = Vector{Matrix{Float64}}(undef, num_obs)
        phasors   = Vector{ComplexF64}(undef, num_obs)

        # Generate noisy signal and extract phasors for each observation
        for k in 1:num_obs
            t_new  = signal_length * test_list[k, 1]
            t_curr = t_curr + t_new
            t      = collect(0.0:dt:t_new)

            # Reconstruct clean signal from true phasors
            sig = (y[k, 1] .* cos.(test_list[k, 2] .* t)) .+
                  (-y[k, 2] .* sin.(test_list[k, 2] .* t))

            # Add Gaussian noise under i.i.d. assumption
            sig_noise = sig .+ data_err .* randn(length(t))

            # Least-squares phasor extraction
            fit          = periodic_ls_fit(t, sig_noise, [test_list[k, 1]])
            data_covs[k] = fit.data_cov
            phasors[k]   = fit.phasors[1]
        end

        # Build block-diagonal inverse data covariance matrix
        R_inv = inv(blockdiag(data_covs...))

        # Pack noisy phasor coefficients into data vector (block layout)
        y_noise                      = zeros(Float64, 2 * num_obs)
        y_noise[1:num_obs]           = real.(phasors)
        y_noise[num_obs+1:2*num_obs] = imag.(phasors)

        # LM inversion
        s_hat, _, _ = grad_inv_lm(y_noise, s_init, fwd_func, obj_func, R_inv, λ, δ)

        # Linearized uncertainty analysis
        unc = param_uncertainty(s_hat, δ, fwd_func, R_inv, model)

        # Accumulate results
        push!(t_save,       t_curr)
        push!(param_stddev_acc, unc.param_sd)
        push!(s_hat_acc,        s_hat)

        t_total       = t_curr
        signal_length = signal_length + 1
    end

    # Convert accumulators to matrices
    param_stddev = reduce(hcat, param_stddev_acc)'
    s_hat_save    = reduce(hcat, s_hat_acc)'

    return SigLenUncResult(t_save, param_stddev, s_hat_save)
end