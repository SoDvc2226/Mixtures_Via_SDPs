using LinearAlgebra, Random, Distributions, Plots

# Step 1: Generate diagonal covariance with given eccentricity
function generate_random_diagonal_covariance(n::Int, ecc::Float64; sigma_min::Float64 = 1.0)
    sigma_max = ecc * sigma_min
    raw = rand(n) .+ 0.1
    sigmas = sigma_min .+ (sigma_max - sigma_min) .* (raw .- minimum(raw)) ./ (maximum(raw) - minimum(raw))
    return Diagonal(sigmas)
end

# Step 2: Generate separated means in [-box_lim, box_lim]^n
function generate_separated_means(K::Int, n::Int, covariances::Vector{<:Diagonal}, c::Float64; box_lim::Float64 = 10.0)
    means = Vector{Vector{Float64}}()
    max_trials = 5000

    for i in 1:K
        found = false
        trial = 0

        while !found && trial < max_trials
            candidate = rand(n) .* (2 * box_lim) .- box_lim

            valid = true
            for j in 1:length(means)
                dist = norm(candidate .- means[j])
                trace_i = sum(diag(covariances[i]))
                trace_j = sum(diag(covariances[j]))
                threshold = c * max(trace_i, trace_j)
                if dist < threshold
                    valid = false
                    break
                end
            end

            if valid
                push!(means, candidate)
                found = true
            else
                trial += 1
            end
        end

        if trial == max_trials
            error("Could not place mean $i after $max_trials trials. Try reducing K or c or increasing box size.")
        end
    end

    return hcat(means...)  # returns an n x K matrix
end

# Step 3: Build full Gaussian Mixture Model
function generate_gmm(K::Int, n::Int; ecc::Float64 = 10.0, c::Float64 = 2.0, box_lim::Float64 = 10.0)
    covariances = [generate_random_diagonal_covariance(n, ecc) for _ in 1:K]
    means = generate_separated_means(K, n, covariances, c; box_lim = box_lim)
    weights = fill(1.0 / K, K)
    return (weights = weights, means = means, covariances = covariances)
end

# Step 4: Plot GMM (only works for 2D)
function plot_gmm(gmm; num_points::Int = 100)
    if size(gmm.means, 1) != 2
        error("plot_gmm only works for 2D data.")
    end

    plt = scatter([], [], title = "Gaussian Mixture", xlabel = "x1", ylabel = "x2",
                  legend = false, aspect_ratio = 1)

    for i in 1:length(gmm.weights)
        mu = gmm.means[:, i]
        Sigma = Matrix(gmm.covariances[i])
        
        # Eigen-decomposition to get ellipse axes
        evals, evecs = eigen(Symmetric(Sigma))
        t = range(0, 2pi, length = num_points)
        circle = [cos.(t)'; sin.(t)']
        
        # Transform unit circle to ellipse
        ellipse = evecs * diagm(sqrt.(evals)) * circle .+ mu

        plot!(plt, ellipse[1, :], ellipse[2, :], lw = 2, alpha = 0.4)
        scatter!(plt, [mu[1]], [mu[2]], marker = (:circle, 4), color = :black)
    end

    return plt
end



using LinearAlgebra, Random, Distributions, Plots

# Generate diagonal covariance matrix with given eccentricity
function generate_random_diagonal_covariance(n::Int, ecc::Float64; sigma_min::Float64 = 1.0)
    sigma_max = ecc * sigma_min
    raw = rand(n) # .+ 0.1
    sigmas = sigma_min .+ (sigma_max - sigma_min) .* (raw .- minimum(raw)) ./ (maximum(raw) - minimum(raw))
    return Diagonal(sigmas)
end

# Generate means in [0,1]^n satisfying the pairwise separation condition
function generate_separated_means(K::Int, n::Int, covariances::Vector{<:Diagonal}, c::Float64)
    means = Vector{Vector{Float64}}()
    max_trials = 100000

    for i in 1:K
        found = false
        trial = 0

        while !found && trial < max_trials
            candidate = rand(n)*2 # uniform in [0,1]^n

            valid = true
            for j in 1:length(means)
                dist = norm(candidate .- means[j])
                trace_i = sum(diag(covariances[i]))
                trace_j = sum(diag(covariances[j]))
                threshold = c * max(trace_i, trace_j)
                if dist < threshold
                    valid = false
                    break
                end
            end

            if valid
                push!(means, candidate)
                found = true
            else
                trial += 1
            end
        end

        if trial == max_trials
            error("Could not place mean $i after $max_trials trials. Try reducing K or c.")
        end
    end

    return hcat(means...)  # n x K
end



function sample_dirichlet_with_min(K::Int, ε::Float64)
    max_trials = 10_000
    for _ in 1:max_trials
        w = rand(Dirichlet(K, 1.0))
        if minimum(w) ≥ ε
            return w
        end
    end
    error("Could not generate Dirichlet sample with min ≥ $ε after $max_trials trials.")
end

function generate_gmm(K::Int, n::Int;
                      ecc::Float64 = 10.0,
                      c::Float64 = 2.0,
                      sigma_min::Float64 = 1.0,
                      weights::Union{Nothing, Vector{Float64}} = nothing)
    
    covariances = [generate_random_diagonal_covariance(n, ecc; sigma_min = sigma_min) for _ in 1:K]
    means = generate_separated_means(K, n, covariances, c)

    if isnothing(weights)
         weights = sample_dirichlet_with_min(K, 0.05)  
    else
        @assert length(weights) == K "Length of weights must match number of components"
        @assert abs(sum(weights) - 1.0) < 1e-6 "Weights must sum to 1"
    end

    return (weights = weights, means = means, covariances = covariances)
end


# Sample N points from the GMM
function sample_from_gmm(gmm, N::Int)
    K = length(gmm.weights)
    n = size(gmm.means, 1)
    X = zeros(n, N)
    labels = zeros(Int, N)

    component_ids = rand(Categorical(gmm.weights), N)

    for i in 1:N
        k = component_ids[i]
        mu = gmm.means[:, k]
        Sigma = Matrix(gmm.covariances[k])
        X[:, i] .= rand(MvNormal(mu, Sigma))
        labels[i] = k
    end

    return X, labels
end
function normalize_data!(X::Matrix{Float64}, gmm::NamedTuple)
    n, N = size(X)
    K = size(gmm.means, 2)

    for i in 1:n
        all_vals = vcat(X[i, :], gmm.means[i, :])
        xmin = minimum(all_vals)
        xmax = maximum(all_vals)

        if xmax == xmin
            X[i, :] .= 0.5
            gmm.means[i, :] .= 0.5
        else
            X[i, :] .= (X[i, :] .- xmin) ./ (xmax - xmin)
            gmm.means[i, :] .= (gmm.means[i, :] .- xmin) ./ (xmax - xmin)
        end
    end
end
# Plot the sampled GMM data (only for 2D)
function plot_gmm(gmm, X, labels)
    if size(X, 1) != 2
        error("plot_gmm only works for 2D data.")
    end
    scatter(X[1, :], X[2, :],
            group = labels,
            title = "Gaussian Mixture (samples)",
            xlabel = "x1", ylabel = "x2",
            #xlims = (0.0, 1.0), ylims = (0.0, 1.0),
            markersize = 2, alpha = 0.5,
            aspect_ratio = 1, legend = false)

    scatter!(gmm.means[1, :], gmm.means[2, :],
             marker = (:circle, 6), color = :black)
end




function generate_multiple_gmms(num::Int, K::Int, n::Int; ecc::Float64 = 10.0, c::Float64 = 2.0, sigma_min::Float64 = 1.0)
    configs = Vector{NamedTuple{(:weights, :means, :covariances), Tuple{Vector{Float64}, Matrix{Float64}, Vector{Diagonal{Float64, Vector{Float64}}}}}}()
    for i in 1:num
        gmm = generate_gmm(K, n; ecc=ecc, c=c, sigma_min=sigma_min)
        push!(configs, gmm)
    end
    return configs
end

# Example usage
gmm_configs = generate_multiple_gmms(20, 3, 2; ecc=5.0, c=1.0, sigma_min=0.01);

function generate_gmm_heteroscedastic(K::Int, n::Int;
                      ecc::Float64 = 10.0,
                      c::Float64 = 2.0,
                      sigma_mins::Union{Float64, Vector{Float64}} = 1.0,
                      weights::Union{Nothing, Vector{Float64}} = nothing)

    # Allow per-component sigma_min (heteroscedastic)
    sigma_min_vec = isa(sigma_mins, Float64) ? fill(sigma_mins, K) : sigma_mins
    @assert length(sigma_min_vec) == K "sigma_mins must have length K"

    covariances = [generate_random_diagonal_covariance(n, ecc; sigma_min = sigma_min_vec[k]) for k in 1:K]
    means = generate_separated_means(K, n, covariances, c)

    if isnothing(weights)
         weights = sample_dirichlet_with_min(K, 0.05)
    else
        @assert length(weights) == K "Length of weights must match number of components"
        @assert abs(sum(weights) - 1.0) < 1e-6 "Weights must sum to 1"
    end

    return (weights = weights, means = means, covariances = covariances)
end
function generate_multiple_gmms_heteroscedastic(num::Int, K::Int, n::Int;
    ecc::Float64 = 10.0, c::Float64 = 2.0, sigma_min_range::Tuple{Float64, Float64} = (0.01, 0.1))

    configs = Vector{NamedTuple{(:weights, :means, :covariances), Tuple{Vector{Float64}, Matrix{Float64}, Vector{Diagonal{Float64, Vector{Float64}}}}}}()
    for _ in 1:num
        sigma_mins = rand(Uniform(sigma_min_range[1], sigma_min_range[2]), K)
        gmm = generate_gmm_heteroscedastic(K, n; ecc=ecc, c=c, sigma_mins=sigma_mins)
        push!(configs, gmm)
    end
    return configs
end
function sample_multiple_gmms(gmms::Vector{T}, N::Int) where T<:NamedTuple
    all_data = Vector{Matrix{Float64}}(undef, length(gmms))
    all_labels = Vector{Vector{Int}}(undef, length(gmms))

    for (i, gmm) in enumerate(gmms)
        X, labels = sample_from_gmm(gmm, N)
        all_data[i] = X
        all_labels[i] = labels
    end

    return all_data, all_labels
end

function normalize_to_unit_box!(X::Matrix{Float64})
    n, N = size(X)
    for i in 1:n
        xmin = minimum(X[i, :])
        xmax = maximum(X[i, :])
        if xmax == xmin
            X[i, :] .= 0.5  # degenerate case, set to mid-point
        else
            X[i, :] .= (X[i, :] .- xmin) ./ (xmax - xmin)
        end
    end
    return X
end


function sorted_empirical_means(X::Matrix{Float64}, labels::Vector{Int})
    n, N = size(X)
    K = maximum(labels)
    means = [zeros(n) for _ in 1:K]
    counts = zeros(Int, K)

    for i in 1:N
        k = labels[i]
        means[k] += X[:, i]
        counts[k] += 1
    end

    for k in 1:K
        means[k] /= counts[k]
    end

    # Sort by the first coordinate
    sorted_means = sort(means, by = x -> x[1])
    return sorted_means
end


function normalize_gmm_and_data(X::Matrix{Float64}, gmm::NamedTuple)
    n, N = size(X)
    K = size(gmm.means, 2)
    
    X_new = copy(X)
    means_new = copy(gmm.means)

    for i in 1:n
        all_vals = vcat(X[i, :], gmm.means[i, :])
        xmin = minimum(all_vals)
        xmax = maximum(all_vals)

        if xmax == xmin
            X_new[i, :] .= 0.5
            means_new[i, :] .= 0.5
        else
            X_new[i, :] .= (X[i, :] .- xmin) ./ (xmax - xmin)
            means_new[i, :] .= (gmm.means[i, :] .- xmin) ./ (xmax - xmin)
        end
    end

    gmm_normalized = (
        weights = gmm.weights,
        means = means_new,
        covariances = gmm.covariances
    )

    return X_new, gmm_normalized
end
