# parameter_space.jl
# Generalised N-dimensional brute force parameter space search.
# Used for objective function visualisation and identifying local minima
# prior to or alongside gradient-based inversion.

"""
parameter_space_search(obj_func, param_vecs...) -> Array{Float64}

Evaluate `obj_func` over an N-dimensional parameter grid.
Returns an N-dimensional array of objective function values matching
the shape of the grid.

# Arguments
- `obj_func`:    function with signature f(Vector{Float64}) -> Float64
- `param_vecs`: any number of parameter vectors defining the grid axes

# Example
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