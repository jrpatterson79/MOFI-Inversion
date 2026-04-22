abstract type AquiferModel end
struct ConfinedAquifer <: AquiferModel end
struct LeakyAquifer    <: AquiferModel end

param_names(::ConfinedAquifer) = ["ln(T [m²/s])", "ln(S [-])"]
param_names(::LeakyAquifer)    = ["ln(T [m²/s])", "ln(S [-])", "ln(L [s⁻¹])"]

num_params(::ConfinedAquifer) = 2
num_params(::LeakyAquifer)    = 3