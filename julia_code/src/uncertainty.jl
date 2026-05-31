# uncertainty.jl
# Linearized uncertainty analysis and confidence region construction.
# Computes parameter covariance, standard deviations, and 2D / 3D confidence
# ellipse/ellipsoid.

# Developed by: Jeremy Patterson
# Created: May 2026 

# The code is provided as open source under the GNU General Public License v3.0. It is provided without warranty, but should perform as described in the manuscript when executed without modification.

using Distributions: Chisq, quantile
using LinearAlgebra 

"""
    UncertaintyResult

Result structure for linearized parameter uncertainty analysis.

# Fields
- `param_cov`: (num_params x num_params) parameter covariance matrix
- `param_sd`:  (num_params,) 95% confidence parameter standard deviations
- `param_CI`:  (num_params x 2) confidence intervals [lower upper]
"""
struct UncertaintyResult
    param_cov :: Matrix{Float64}
    param_sd  :: Vector{Float64}
    param_CI  :: Matrix{Float64}
end

# Linearized error propagation
"""
    param_uncertainty(s_opt, δ, fwd_func, R_inv) -> UncertaintyResult

Compute linearized parameter uncertainty from the Jacobian and data covariance matrix.

# Arguments
- `s_opt`:    (num_params,) optimal parameter vector (log-space)
- `δ`:        (num_params,) finite difference perturbations for Jacobian
- `fwd_func`: forward model closure with signature f(s) -> Vector{Float64}
- `R_inv`:    (num_data x num_data) inverse data error covariance matrix

# Returns
`UncertaintyResult` with fields `param_cov`, `param_sd`, and `param_CI`
"""
function param_uncertainty(
    s_opt    :: Vector{Float64},
    δ        :: Vector{Float64},
    fwd_func :: Function,
    R_inv    :: Matrix{Float64}
)
    J         = sens_mat(s_opt, δ, fwd_func)
    param_cov = inv(J' * R_inv * J)
    param_sd  = 1.96 .* sqrt.(diag(param_cov))
    param_CI  = [s_opt .- param_sd  s_opt .+ param_sd]

    return UncertaintyResult(param_cov, param_sd, param_CI)
end

"""
    ConfidenceRegion

Result structure for the linearized confidence region boundary.

# Fields
- `points`:       (num_points x num_params) coordinates of the ellipse/ellipsoid
                  surface at the 95% confidence level
- `e_vec_scaled`: (num_params x num_params) scaled eigenvectors defining the
                  ellipse/ellipsoid axes
"""
struct ConfidenceRegion
    points       :: Matrix{Float64}     # ellipse/ellipsoid coordinate points
    e_vec_scaled :: Matrix{Float64}     # scaled eigenvector (ellipsoid axes)
end

# Error ellipse calculation
"""
    confidence_region(param_cov, s_opt, model::AquiferModel) -> Matrix{Float64}

Compute the linearized confidence region boundary for estimated parameters.
Returns a (num_points x num_params) matrix of points on the ellipse/ellipsoid
surface at the 95% confidence level.

# Arguments
- `param_cov`: (num_params x num_params) parameter covariance matrix
- `s_opt`:     (num_params,) optimal parameter vector (log-space)

# Returns
`ConfidenceRegion` with fields `points` and `e_vec_scaled`
"""
function confidence_region(
    param_cov :: Matrix{Float64},
    s_opt     :: Vector{Float64}
)
    num_params = length(s_opt)

    # Eigendecomposition — works for any dimension
    e_val, e_vec = eigen(inv(param_cov))

    # Chi-squared scaling — degrees of freedom = num_params
    Δ = sqrt(quantile(Chisq(num_params), 0.95))

    # Scaled radius along each eigenvector axis
    r_ell = Δ ./ sqrt.(e_val)

    # Generate unit surface points for the appropriate dimension
    unit_surface = _unit_surface(param_cov)

    # Rotate and scale unit surface into parameter space
    e_vec_scaled = e_vec * diagm(r_ell)
    points       = (e_vec_scaled * unit_surface')' .+ s_opt'

    return ConfidenceRegion(points, e_vec_scaled)
end

# ── Unit surface generation ───────────────────────────────────────────────────

function _unit_surface(param_cov::Matrix{Float64})
    n = size(param_cov, 1)

    # Confined uncertainty ellipse
    if n == 2
        θ = range(0, 2π, length=500)
        return [cos.(θ) sin.(θ)]
    
    # Leaky uncertainty ellipse
    elseif n == 3
        n = 100
        θ = range(0,  π, length=n)
        φ = range(0, 2π, length=n)
        Θ = repeat(θ,  1, n)
        Φ = repeat(φ', n, 1)
        x = vec(sin.(Θ) .* cos.(Φ))
        y = vec(sin.(Θ) .* sin.(Φ))
        z = vec(cos.(Θ))
        return [x y z]
    else
        error("_unit_surface: unsupported dimensionality $n")
    end
end