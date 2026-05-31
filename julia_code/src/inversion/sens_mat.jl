# sens_mat.jl
# Model-agnostic forward finite-difference Jacobian function

# Developed by: Jeremy Patterson
# Created: May 2026 

# The code is provided as open source under the GNU General Public License v3.0. It is provided without warranty, but should perform as described in the manuscript when executed without modification.

"""
    sens_mat(s, δ, fwd_model_func) -> Matrix{Float64}

Compute the parameter sensitivity (Jacobian) matrix via forward finite
differences. Model agnostic: operates on any forward model function that maps a parameter vector to a data vector.

# Arguments
- `s`:             (num_params,) parameter vector (log-space)
- `δ`:             (num_params,) finite difference perturbations
- `fwd_model_func`: forward model closure with signature f(s) -> Vector{Float64}

# Returns
- `J`: (num_data x num_params) Jacobian matrix
"""
function sens_mat(
    s::Vector{Float64}, 
    δ::Vector{Float64}, 
    fwd_model_func::Function
)

    base_vals  = fwd_model_func(s)
    num_data   = length(base_vals)
    num_params = length(s)

    J = zeros(Float64, num_data, num_params)

    for i in 1:num_params
        s_pert    = copy(s)   # copy, not reference
        s_pert[i] = s[i] + δ[i]

        mod_vals  = fwd_model_func(s_pert)
        J[:, i]   = (mod_vals - base_vals) ./ δ[i]
    end

    return J
end