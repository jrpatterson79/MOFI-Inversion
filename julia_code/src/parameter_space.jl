# parameter_space.jl
# Generalised N-dimensional brute force parameter space search.
# Used for objective function visualisation and identifying local minima
# prior to or alongside gradient-based inversion.

# Developed by: Jeremy Patterson
# Created: May 2026 

# The code is provided as open source under the GNU General Public License v3.0. It is provided without warranty, but should perform as described in the manuscript when executed without modification.

"""
    parameter_space_search(obj_func, param_vecs...) -> Array{Float64}

Evaluate `obj_func` over an N-dimensional parameter grid using
multithreading. Returns an N-dimensional array of objective function
values matching the shape of the grid.

# Arguments
- `obj_func`:    objective function closure with signature f(Vector{Float64}) -> Float64
- `param_vecs`: any number of parameter vectors defining the grid axes

# Returns
- N-dimensional `Array{Float64}` of objective values

# Examples
    # Confined (2D)
    mod_norm = parameter_space_search(obj_func, T_vec, S_vec)

    # Leaky (3D)
    mod_norm = parameter_space_search(obj_func, T_vec, S_vec, L_vec)
"""
function parameter_space_search(obj_func::Function, param_vecs...)
    grid    = collect(Iterators.product(param_vecs...))
    results = similar(grid, Float64)

    Threads.@threads for idx in eachindex(grid)
        results[idx] = obj_func(collect(grid[idx]))
    end

    return results
end