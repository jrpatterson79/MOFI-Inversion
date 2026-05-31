# forward_model.jl

# Developed by: Jeremy Patterson
# Created: May 2026 

# The code is provided as open source under the GNU General Public License v3.0. It is provided without warranty, but should perform as described in the manuscript when executed without modification.

using SpecialFunctions: besselk

"""
    rasmussen_phasor(test_list, s, model) -> Vector{Float64}

Compute the steady-periodic phasor head response at an observation well
using the Rasmussen et al. (2003) analytical solution. Dispatches on
`model` to select the confined or leaky aquifer formulation.

# Arguments
- `test_list`: (num_obs x 4) matrix [Period(s), ω(rad/s), Q_max(m³/s), r(m)]
- `s`:         (num_params,) parameter vector (log-space)
- `model`:     `ConfinedAquifer()` or `LeakyAquifer()`

# Returns
- `y_mod`: (2*num_obs,) vector of phasor coefficients packed as
           [real(h₁)…real(hₙ), imag(h₁)…imag(hₙ)]

# Reference
Rasmussen, T.C., et al. (2003). Applying Oscillatory Flow to Determine
Aquifer Characteristics. Vadose Zone Journal, 2(4), 514-521.
"""
function rasmussen_phasor(test_list::Matrix{Float64}, s::Vector{Float64}, model::AquiferModel)
    s_phys = exp.(s)    # log-space is a hard contract, always transform unconditionally
    _compute_phasors(test_list, s_phys, model)
end

function _compute_phasors(test_list::Matrix{Float64}, s::Vector{Float64}, ::ConfinedAquifer)
    T, S    = s[1], s[2]
    D       = T / S
    ω       = test_list[:, 2]
    Q_max   = test_list[:, 3]
    r       = test_list[:, 4]
    num_obs = length(r)

    phasor = map(1:num_obs) do j
        arg = sqrt((1im * r[j]^2 * ω[j]) / D)
        Q_max[j] / (2π * T) * besselk(0, arg)
    end

    _pack_phasors(phasor)
end

function _compute_phasors(test_list::Matrix{Float64}, s::Vector{Float64}, ::LeakyAquifer)
    T, S, L = s[1], s[2], s[3]
    D       = T / S
    B_sq    = T / L
    ω       = test_list[:, 2]
    Q_max   = test_list[:, 3]
    r       = test_list[:, 4]
    num_obs = length(r)

    phasor = map(1:num_obs) do j
        arg = sqrt((1im * r[j]^2 * ω[j]) / D + r[j]^2 / B_sq)
        Q_max[j] / (2π * T) * besselk(0, arg)
    end

    _pack_phasors(phasor)
end

function _pack_phasors(phasor::Vector{ComplexF64})
    n     = length(phasor)
    y_mod = zeros(Float64, 2n)
    y_mod[1:n]    = real.(phasor)
    y_mod[n+1:2n] = imag.(phasor)
    return y_mod
end