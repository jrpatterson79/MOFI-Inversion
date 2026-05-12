# periodic_ls_fit.jl
# Least-squares fitting of periodic signals to extract phasor coefficients.
# Constructs a design matrix of sinusoids at specified periods, solves for
# optimal coefficients, and returns complex phasor representations along
# with the data covariance matrix for use in weighted inversion.

struct LsFitResult
    data_cov :: Matrix{Float64}
    phasors  :: Vector{ComplexF64}
    mse      :: Float64
end

function periodic_ls_fit(times::Vector{Float64}, sig::Vector{Float64}, periods::Vector{Float64})

    length(sig) == length(times) || error("Length of sig must match length of times")

    num_times = length(times)
    num_per   = length(periods)

    # Build design matrix
    G = zeros(Float64, num_times, 2 * num_per)
    for (p, P) in enumerate(periods)
        col         = 2p - 1
        G[:, col]   =  cos.(2π / P .* times)
        G[:, col+1] = -sin.(2π / P .* times)
    end

    # Least-squares solution
    GtG   = G' * G
    c_opt = GtG \ (G' * sig)

    # Mean squared error and data covariance matrix
    residual = sig - G * c_opt
    mse      = (residual' * residual) / num_times
    data_cov = inv(GtG) .* mse

    # Pack into complex phasors
    phasors = map(1:num_per) do p
        col = 2p - 1
        c_opt[col] + 1im * c_opt[col + 1]
    end

    return LsFitResult(data_cov, phasors, mse)
end