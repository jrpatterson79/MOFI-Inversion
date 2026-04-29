# forward_model.jl
# Steady-periodic analytical forward model for oscillatory pumping tests.
# Computes complex phasor head response at an observation well using the
# Rasmussen et al. (2003) solution for confined and leaky aquifer conditions.
#
# Dispatch on AquiferModel type routes to the appropriate formulation —
# ConfinedAquifer uses hydraulic diffusivity only, LeakyAquifer adds a
# leakage term to the Bessel function argument.
#
# Reference:
#   Rasmussen, T.C., et al. (2003). Applying Oscillatory Flow to Determine
#   Aquifer Characteristics. Vadose Zone Journal, 2(4), 514-521.

using SpecialFunctions: besselk

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