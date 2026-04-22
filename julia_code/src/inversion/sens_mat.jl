function sens_mat(s::Vector{Float64}, δ::Vector{Float64}, fwd_model_func::Function)
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