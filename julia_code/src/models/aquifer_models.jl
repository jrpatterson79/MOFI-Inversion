#aquifer_models.jl

# Developed by: Jeremy Patterson
# Created: May 2026 

# The code is provided as open source under the GNU General Public License v3.0. It is provided without warranty, but should perform as described in the manuscript when executed without modification.

"""
    AquiferModel

Abstract supertype for aquifer model formulations.
Subtypes: `ConfinedAquifer`, `LeakyAquifer`.
"""
abstract type AquiferModel end

"""
    ConfinedAquifer <: AquiferModel

Confined aquifer model. Parameterized by transmissivity T and storativity S.
"""
struct ConfinedAquifer <: AquiferModel end

"""
    LeakyAquifer <: AquiferModel

Leaky aquifer model. Parameterized by transmissivity T, storativity S,
and leakage coefficient L.
"""
struct LeakyAquifer    <: AquiferModel end

param_names(::ConfinedAquifer) = ["ln(T [m²/s])", "ln(S [-])"]
param_names(::LeakyAquifer)    = ["ln(T [m²/s])", "ln(S [-])", "ln(L [s⁻¹])"]

num_params(::ConfinedAquifer) = 2
num_params(::LeakyAquifer)    = 3