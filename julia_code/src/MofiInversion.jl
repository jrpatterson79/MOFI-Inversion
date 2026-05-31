# MofiInversion.jl
# Main module for the mofi-inversion package.
#
# Developed by: Jeremy Patterson
# Created: May 2026 

# The code is provided as open source under the GNU General Public License v3.0. It is provided without warranty, but should perform as described in the manuscript when executed without modification.

"""
    MofiInversion

Multi-frequency oscillatory flow test (MOFI) inversion for aquifer
characterization using the steady-periodic analytical solutions of
Rasmussen et al. (2003).

# Module structure
- `models/`            → aquifer type hierarchy and forward model
- `inversion/`         → Levenberg-Marquardt inversion and Jacobian
- `signal_processing/` → phasor extraction and signal length uncertainty
- `uncertainty.jl`     → linearized covariance and confidence regions
- `parameter_space.jl` → N-dimensional brute force grid search

# References
Rasmussen, T.C., et al. (2003). Applying Oscillatory Flow to Determine
Aquifer Characteristics. Vadose Zone Journal, 2(4), 514-521.

Patterson, J.R. & Cardiff, M. (2022). Aquifer Characterization and
Uncertainty in Multi-Frequency Oscillatory Flow Tests: Approach and
Insights. Groundwater, 60(2), 180-191. https://doi.org/10.1111/gwat.13134
"""
module MofiInversion

    # Models
    include("models/aquifer_models.jl")
    include("models/forward_model.jl")

        # Signal Processing
    include("signal_processing/periodic_ls_fit.jl")
    include("signal_processing/sig_len_unc.jl")

    # Inversion
    include("inversion/sens_mat.jl")
    include("inversion/grad_inv_lm.jl")

    # Analysis
    include("uncertainty.jl")
    include("parameter_space.jl")

    # Public API — internal helpers (e.g. _pack_phasors) are intentionally excluded
    export  # Models
        ConfinedAquifer, LeakyAquifer,
        param_names, num_params,
        # Forward Model
        rasmussen_phasor,
        # Inversion
        InversionResult, 
        grad_inv_lm, sens_mat,
        # Signal Processing
        LsFitResult, 
        periodic_ls_fit, sig_len_unc, build_R_inv,
        # Uncertainty
        ConfidenceRegion, UncertaintyResult,
        param_uncertainty, confidence_region,
        # Parameter Space
        parameter_space_search
end