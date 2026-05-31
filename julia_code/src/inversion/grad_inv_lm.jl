#grad_inv_lm.jl
# Gradient-based Levenberg-Marquardt algorithm that minimizes the misfit between measured modeled values.

# Developed by: Jeremy Patterson
# Created: May 2026 

# The code is provided as open source under the GNU General Public License v3.0. It is provided without warranty, but should perform as described in the manuscript when executed without modification.

using LinearAlgebra
using Optim

"""
    InversionResult

Result structure for Levenberg-Marquardt inversion.

# Fields
- `s_curr`:    (num_params,) optimal parameter vector (log-space)
- `s_update`:  (num_iter x num_params) parameter iterates at each step
- `model_err`: final objective function value
- `flag`:      convergence flag — 1 = converged, 0 = max iterations exceeded
"""
struct InversionResult
    s_curr     :: Vector{Float64}
    s_update   :: Matrix{Float64}
    model_err  :: Float64
    flag       :: Int
end

"""
    grad_inv_lm(y, s, fwd_model_func, obj_func, λ_init, δ) -> InversionResult

Non-linear gradient inversion using the Levenberg-Marquardt algorithm.

# Arguments
- `y`:             (2*num_obs,) vector of phasor coefficients (calibration data)
- `s`:             (num_params,) initial parameter guess (log-space)
- `fwd_model_func`: forward model closure with signature f(s) -> Vector{Float64}
- `obj_func`:      objective function closure with signature f(s) -> Float64
- `λ_init`:        initial LM regularization parameter
- `δ`:             (num_params,) finite difference perturbations for Jacobian

# Keyword Arguments
- `max_iter`:  maximum number of iterations (default: 75)
- `obj_close`: relative objective function change convergence tolerance (default: 1e-6)
- `s_close`:   relative parameter change convergence tolerance (default: 1e-6)

# Returns
`InversionResult` with fields `s_curr`, `s_update`, `model_err`, `flag`
"""
function grad_inv_lm(y, s, fwd_model_func, obj_func, λ_init, δ)

    # Convergence criteria
    max_linevals = 100
    obj_close    = 1e-6
    s_close      = 1e-6
    max_iter     = 75

    # Initialise
    s_curr    = copy(s)
    λ         = float(λ_init)
    iter      = 0
    num_param = length(s_curr)

    s_update = Matrix{Float64}(undef, max_iter, num_param)   # pre-allocate

    while iter < max_iter

        # Forward model at current parameters
        y_curr = fwd_model_func(s_curr)

        # Jacobian (sensitivity matrix)
        J_tilde = sens_mat(s_curr, δ, fwd_model_func)

        # Levenberg-Marquardt step
        step = -(J_tilde' * J_tilde + λ * I(num_param)) \ (-J_tilde' * (y_curr - y))

        # Line search along LM step direction
        step_obj  = alpha -> obj_func(s_curr .+ alpha .* step)
        result     = Optim.optimize(step_obj, [0.5], NelderMead())
        alpha_best = Optim.minimizer(result)[1]

        # Candidate update
        s_new    = s_curr .+ alpha_best .* step
        s_change = maximum(abs.((s_new .- s_curr) ./ s_curr))

        # Objective values (log10 scale for relative change)
        obj_curr_val  = log10(obj_func(s_curr))
        obj_new_val   = log10(obj_func(s_new))
        obj_change    = abs((obj_curr_val - obj_new_val) / obj_curr_val)

        println("iter=$iter  s_change=$s_change  obj_change=$obj_change  λ=$λ")

        # Convergence check
        if obj_change <= obj_close && s_change <= s_close
            model_err = obj_func(s_curr)
            out_flag = 1
            return InversionResult(s_curr, s_update[1:iter, :], model_err, out_flag)
        end

        # Store current iterate
        s_update[iter + 1, :] = s_curr

        if obj_new_val < obj_curr_val
            # Accept step, relax regularisation
            s_curr = s_new
            λ = max(λ * 1e-1, 1e-12)
        else
            # Reject step, tighten regularisation
            λ = min(λ * 1e1, 1e10)
        end

        iter += 1
    end

    # Max iterations reached
    model_err = obj_func(s_curr)
    out_flag  = 0
    @warn "grad_inv_lm: maximum iterations ($max_iter) exceeded"
    return InversionResult(s_curr, s_update[1:iter, :], model_err, out_flag)
end