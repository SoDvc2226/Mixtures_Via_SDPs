using JuMP
using MosekTools
using DynamicPolynomials
using MultivariatePolynomials
using TSSOS
using LinearAlgebra, Random, Plots, Distributions, IterTools, Combinatorics



####################################################################################################
###################################################    JOINT CASE ################################################
####################################################################################################


function generate_gaussian_mixtures(k, means, variances, mixing_p; seed=1, n_samples=500)
    @assert length(means) == k
    @assert length(variances) == k
    @assert length(mixing_p) == k
    @assert isapprox(sum(mixing_p), 1.0; atol=1e-6) "Mixing probabilities must sum to 1."

    dists = [MvNormal(means[i], variances[i]) for i in 1:k]
    mix = Categorical(mixing_p)

    d = length(means[1])
    samples = zeros(n_samples, d)
    labels = zeros(Int, n_samples)  

    Random.seed!(seed)
    for i in 1:n_samples
        comp = rand(mix)
        samples[i, :] = rand(dists[comp])
        labels[i] = comp
    end

    return samples, labels
end



function get_monomial_elements(monomial)
    """
    Extracts the individual elements (variables) of a given monomial, accounting for their powers. 
    Each variable in the monomial appears in the output list as many times as its exponent.
    
    # Arguments
    - `monomial`: A symbolic monomial expression
    
    # Returns
    - `terms::Vector`: A vector containing the variables in the monomial, repeated according to their powers.
    
    # Example
    # Suppose `x^2 * y` is a monomial:
    monomial = :(x^2 * y)
    get_monomial_elements(monomial) 
    # Output: [x, x, y]
    """
    terms = []
    for term in unique(variables(monomial))
        exp = degree(monomial, term)
        append!(terms, repeat([term], exp))
    end
    return terms
end

function extract_exponents(monomials, vars)
    """
    Extracts exponent vectors from a list of monomials using `get_monomial_elements`.

    # Arguments
    - `monomials`: Vector of monomials.
    - `vars`: Vector of variables `[x1, x2, ..., xn]` defining the ordering.

    # Returns
    - `Vector{Vector{Int}}`: A list of exponent vectors corresponding to each monomial.
    """
    exponent_vectors = []

    for monomial in monomials
        exp_vector = zeros(Int, length(vars))

        monomial_elements = get_monomial_elements(monomial)

        for (i, var) in enumerate(vars)
            exp_vector[i] = count(x -> x == var, monomial_elements)
        end

        push!(exponent_vectors, exp_vector)
    end

    return exponent_vectors
end

function double_factorial(n)
    return n <= 0 ? 1 : factorial(n) ÷ (2^(n ÷ 2) * factorial(n ÷ 2))
end
function generate_multi_indices(n::Int, order::Int)
    return [collect(p) for p in Iterators.filter(p -> sum(p) == order, Iterators.product(fill(0:order, n)...))]
end

function compute_empirical_moments_from_basis(n::Int, samples::Matrix, vb, vars)
    """
    Computes empirical moments for the monomials given in `vb`.

    Arguments:
    - `n`: Number of variables.
    - `samples`: Matrix where each row is a sample and columns correspond to variables.
    - `vb`: Vector of monomials forming the basis.
    - `vars`: Vector of variables `[x1, x2, ..., xn]`.

    Returns:
    - `Vector`: Empirical moments computed in the order of `vb`.
    """
    N = size(samples, 1)  
    multi_indices = extract_exponents(vb, vars)  
    moments = Vector{Float64}(undef, length(vb))  

    for (i, alpha) in enumerate(multi_indices)
        moment_value = sum(prod(samples[row, j]^alpha[j] for j in 1:n) for row in 1:N) / N
        moments[i] = moment_value
    end

    return moments
end


function compute_empirical_moments(n::Int, order::Int, samples::Matrix)
    N = size(samples, 1)  # Number of samples
    moments = Dict()  # Dictionary to store computed empirical moments

    alphas = generate_multi_indices(n, order)

    for alpha in alphas
       
        moment_value = sum(prod(samples[row, i]^alpha[i] for i in 1:n) for row in 1:N) / N
        moments[alpha] = moment_value
    end

    return moments  
end






function gaussian_moments_by_order(n::Int, order::Int, m::Vector, Sigma::Vector)
    @assert length(m) == n "Mean vector must match dimension n"
    @assert length(Sigma) == n "Variance vector must match dimension n"

    moments = Dict()  # Dictionary to store computed moments

    alphas = generate_multi_indices(n, order)

    for alpha in alphas
        moment = zero(m[1])  

        even_indices = [collect(0:2:alpha[i]) for i in 1:n]

        for k in Iterators.product(even_indices...)
            k_vec = collect(k)  

            binom_terms = prod(binomial(alpha[i], k_vec[i]) for i in 1:n)

            m_vec = k_vec .÷ 2

            double_fact_terms = prod(double_factorial(2 * m_vec[i] - 1) for i in 1:n)

            mean_terms = prod(m[i]^(alpha[i] - k_vec[i]) for i in 1:n)

            variance_terms = prod(Sigma[i]^(2 * m_vec[i]) for i in 1:n)  # FIX: Sigma[i] now appears as σ_i^2

            term = binom_terms * mean_terms * double_fact_terms * variance_terms

            moment += term
        end

        moments[alpha] = moment  
    end

    return moments  
end




function compute_theoretical_moments(n::Int, vb, vars, m::Vector, Sigma::Vector)
    """
    Computes theoretical Gaussian moments for a given monomial basis `vb`.

    Arguments:
    - `n`: Number of variables.
    - `vb`: Vector of monomials forming the basis.
    - `vars`: Vector of variables `[x1, x2, ..., xn]`.
    - `m`: Vector of means `[m1, ..., mn]`.
    - `Sigma`: Vector of standard deviations `[σ1, ..., σn]`.

    Returns:
    - `Vector{Any}`: Theoretical moments computed in the same order as `vb`.
    """
    @assert length(m) == n "Mean vector must match dimension n"
    @assert length(Sigma) == n "Variance vector must match dimension n"

    # Extract multi-indices (exponents) from vb
    alphas = extract_exponents(vb, vars)
    
    theoretical_moments = Vector{Any}(undef, length(vb))  

    for (i, alpha) in enumerate(alphas)
        moment = zero(m[1])  
        
       
        even_indices = [collect(0:2:alpha[j]) for j in 1:n]

        for k in Iterators.product(even_indices...)
            k_vec = collect(k)  

            # Compute binomial coefficients
            binom_terms = prod(binomial(alpha[j], k_vec[j]) for j in 1:n)

            # Compute m_i values
            m_vec = k_vec .÷ 2

            # Compute double factorial terms symbolically
            double_fact_terms = prod(double_factorial(2 * m_vec[j] - 1) for j in 1:n)

            # Compute mean terms symbolically
            mean_terms = prod(m[j]^(alpha[j] - k_vec[j]) for j in 1:n)

            # Compute variance terms (since the covariance matrix is diagonal)
            variance_terms = prod(Sigma[j]^(2 * m_vec[j]) for j in 1:n)  

            # Compute the final symbolic term for this k-vector
            term = binom_terms * mean_terms * double_fact_terms * variance_terms

            moment += term
        end

        theoretical_moments[i] = moment  
    end

    return theoretical_moments  
end


function create_SOS_model(n, d, m, sigma, S, samples)

    
model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)

    
@polyvar x[1:n]
@polyvar y[1:n]
v, vc, vb = add_poly!(model, x, 2d)
w, wc, wb = add_poly!(model, y, 2d)

@polyvar q[1:binomial(n+2d,2d)]
@polyvar g[1:binomial(n+2d,2d)]


em_mom=compute_empirical_moments_from_basis(n, samples, vb, [x[i] for i=1:n])

obj_f=dot(em_mom,vc)


sos1=dot(x-y,x-y)-dot(vc,vb)-dot(wc,wb)
model,info1 = add_psatz!(model, sos1, [x;y], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr1")


gauss_moments=compute_theoretical_moments(n, wb, [y[i] for i=1:n], m, sigma)


sos2=sum(wc[i]*gauss_moments[i] for i=1:length(wc))+0.001*sum(m[i]^2 for i=1:length(m))   # Trace of the moment matrix penalization
model,info2 = add_psatz!(model, sos2, [m;sigma], S, [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr2")

@objective(model, Max, obj_f)
optimize!(model)
objv = objective_value(model)
@show objv

    # retrieve moment matrices
moment1 = [-dual(constraint_by_name(model, "constr1[$i]")) for i=1:size(info1.tsupp, 2)]
MomMat1 = get_moment_matrix(moment1, info1.tsupp, info1.cql, info1.basis)

moment2 = [-dual(constraint_by_name(model, "constr2[$i]")) for i=1:size(info2.tsupp, 2)]
MomMat2 = get_moment_matrix(moment2, info2.tsupp, info2.cql, info2.basis)


return objv, MomMat1, MomMat2
    
    
end






function create_SOS_model_np(n, d, m, sigma, S, samples)

    
model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)

    
@polyvar x[1:n]
@polyvar y[1:n]
q, qc, qb = add_poly!(model, x, 2d)
g, gc, gb = add_poly!(model, y, 2d)

em_mom=compute_empirical_moments_from_basis(n, samples, qb, [x[i] for i=1:n])

obj_f=dot(em_mom,qc)


sos1=dot(x-y,x-y)-dot(qc,qb)-dot(gc,gb)
model,info1 = add_psatz!(model, sos1, [x;y], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr1")


gauss_moments=compute_theoretical_moments(n, gb, [y[i] for i=1:n], m, sigma)


sos2=sum(gc[i]*gauss_moments[i] for i=1:length(gc))   # Trace of the moment matrix penalization
model,info2 = add_psatz!(model, sos2, [m;sigma], S, [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr2")

@objective(model, Max, obj_f)
optimize!(model)
objv = objective_value(model)
@show objv

    
# retrieve moment matrices
moment1 = [-dual(constraint_by_name(model, "constr1[$i]")) for i=1:size(info1.tsupp, 2)]
MomMat1 = get_moment_matrix(moment1, info1.tsupp, info1.cql, info1.basis)

moment2 = [-dual(constraint_by_name(model, "constr2[$i]")) for i=1:size(info2.tsupp, 2)]
MomMat2 = get_moment_matrix(moment2, info2.tsupp, info2.cql, info2.basis)


return objv, MomMat1, MomMat2, model, [info1,info2]
    
    
end





# 1) Moments up to a maximum order (0..kmax)
"""
    empirical_moments_1d(x::AbstractVector{<:Real}, kmax::Integer;
                         include_zero::Bool = true)

Return the vector of raw empirical moments m where
m[r+1] = mean(x.^r) for r = 0..kmax (if `include_zero=true`), otherwise r=1..kmax.
Assumes nonnegative integer orders.
"""
function empirical_moments_1d(x::AbstractVector{<:Real}, kmax::Integer;
                              include_zero::Bool = true)
    @assert kmax ≥ 0 "kmax must be ≥ 0"
    n = length(x)
    @assert n > 0 "x must be nonempty"
    x = collect(float.(x))

    # Efficient accumulation: for each xi, build 1, xi, xi^2, ...
    m = zeros(kmax + 1)
    @inbounds for xi in x
        p = 1.0
        m[1] += p                    # r = 0
        for r in 1:kmax              # r = 1..kmax
            p *= xi
            m[r+1] += p
        end
    end
    m ./= n
    return include_zero ? m : m[2:end]
end

# 2) Moments for an explicit list of orders (e.g., [0,2,5])
"""
    empirical_moments_1d(x::AbstractVector{<:Real}, orders::AbstractVector{<:Integer})

Return mean(x.^r) for each r in `orders` (raw moments). Orders must be ≥ 0.
"""
function empirical_moments_1d(x::AbstractVector{<:Real}, orders::AbstractVector{<:Integer})
    @assert all(≥(0), orders) "orders must be nonnegative integers"
    x = collect(float.(x))
    return [mean(x .^ r) for r in orders]
end






function slack_create_SOS_model(n, d, m, sigma, S, samples,eps_eq,eps_tr; tr_reg=false)   #I have to indicate positivity of g_j^+ and g_j-

    
model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)

    
@polyvar x[1:n]
@polyvar y[1:n]
q, qc, qb = add_poly!(model, x, 2d)   #vc are coefficients of q
gp, gpc, gpb = add_poly!(model, y, 2d)    # coefficients of g_j^+
gm, gmc, gmb = add_poly!(model, y, 2d)    # coefficients of g_j^-

if n==1
    em_mom=compute_empirical_moments_from_basis(1, reshape(samples, :, 1), vb, vars)
else

    em_mom=compute_empirical_moments_from_basis(n, samples, qb, [x[i] for i=1:n])
end

obj_f=dot(em_mom,qc)-eps_eq*sum(gpc[i]+gmc[i] for i=1:length(gpc))


#sos1=dot(x-y,x-y)-dot(qc,qb)-dot(gpc-gmc,gpb) # Old version
sos1=dot(x-y,x-y)-dot(qc,qb)-dot(gmc-gpc,gpb)         # New version
model,info1 = add_psatz!(model, sos1, [x;y], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr1")


gauss_moments=compute_theoretical_moments(n, gpb, [y[i] for i=1:n], m, sigma)

if tr_reg 
    #sos2=sum((gpc[i]-gmc[i])*gauss_moments[i] for i=1:length(gpc)) + epss*sum(m[i]^2 for i=1:length(m))   # Trace of the moment matrix penalization kinda
    #sos2=sum((gpc[i]-gmc[i])*gauss_moments[i] for i=1:length(gpc)) + eps_tr*trace_like_expression(m,d)         # Old one
    sos2=sum(-(gpc[i]-gmc[i])*gauss_moments[i] for i=1:length(gpc)) + eps_tr*trace_like_expression(m,d)         # New one
else
    sos2=sum(-(gpc[i]-gmc[i])*gauss_moments[i] for i=1:length(gpc))
end
        
model,info2 = add_psatz!(model, sos2, [m;sigma], S, [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr2")


# Impose nonnegativity
for i in 1:length(gpc)
    @constraint(model, gpc[i] ≥ 0)
end

for i in 1:length(gmc)
    @constraint(model, gmc[i] ≥ 0)
end

#println(model)  Check this tomorrow

@objective(model, Max, obj_f)
optimize!(model)
objv = objective_value(model)
@show objv

# retrieve moment matrices
moment1 = [-dual(constraint_by_name(model, "constr1[$i]")) for i=1:size(info1.tsupp, 2)]
MomMat1 = get_moment_matrix(moment1, info1.tsupp, info1.cql, info1.basis)

moment2 = [-dual(constraint_by_name(model, "constr2[$i]")) for i=1:size(info2.tsupp, 2)]
MomMat2 = get_moment_matrix(moment2, info2.tsupp, info2.cql, info2.basis)

#qc_opt = value.(qc)
#gpc_opt = value.(gpc)
#gmc_opt = value.(gmc)

# retrieve Gram matrices
GramMat = Vector{Vector{Vector{Union{Float64,Matrix{Float64}}}}}(undef, info1.cql)
for i = 1:info1.cql
    GramMat[i] = Vector{Vector{Union{Float64,Matrix{Float64}}}}(undef, 1+length(info1.I[i])+length(info1.J[i]))
    for j = 1:1+length(info1.I[i])+length(info1.J[i])
        GramMat[i][j] = [value.(info1.gram[i][j][l]) for l = 1:info1.cl[i][j]]
    end
end

GramMat2 = Vector{Vector{Vector{Union{Float64,Matrix{Float64}}}}}(undef, info2.cql)
for i = 1:info2.cql
    GramMat2[i] = Vector{Vector{Union{Float64,Matrix{Float64}}}}(undef, 1+length(info2.I[i])+length(info2.J[i]))
    for j = 1:1+length(info2.I[i])+length(info2.J[i])
        GramMat2[i][j] = [value.(info2.gram[i][j][l]) for l = 1:info2.cl[i][j]]
    end
end

    
    

    

return objv, MomMat1, MomMat2,  model, [info1,info2] #GramMat, GramMat2
    
    
end






function multivariate_Gaussian_W2(n, d, m, sigma, S, samples, trace_penalization, vareps)  

    
model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)

    
@polyvar x[1:n]
@polyvar y[1:n]
q, qc, qb = add_poly!(model, x, 2d)   # coefficients of q
g, gc, gb = add_poly!(model, y, 2d)    # coefficients of g



em_mom=compute_empirical_moments_from_basis(n, samples, qb, [x[i] for i=1:n])

obj_f=dot(em_mom,qc)


sos1=dot(x-y,x-y)-dot(qc,qb)+dot(gc,gb)         
model,info1 = add_psatz!(model, sos1, [x;y], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr1")


gauss_moments=compute_theoretical_moments(n, gb, [y[i] for i=1:n], m, sigma)

if trace_penalization 
    #sos2= vareps*trace_like_expression(m, d) - sum(gc[i]*gauss_moments[i] for i=1:length(gc))
    sos2= vareps*(trace_penalty(m, sigma, d)+trace_like_expression(m,d))  - sum(gc[i]*gauss_moments[i] for i=1:length(gc))       
else
    sos2= - sum(gc[i]*gauss_moments[i] for i=1:length(gc))
end
        
model,info2 = add_psatz!(model, sos2, [m;sigma], S, [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr2")

@objective(model, Max, obj_f)
optimize!(model)
objv = objective_value(model)
@show objv

# retrieve moment matrices
moment1 = [-dual(constraint_by_name(model, "constr1[$i]")) for i=1:size(info1.tsupp, 2)]
MomMat1 = get_moment_matrix(moment1, info1.tsupp, info1.cql, info1.basis)

moment2 = [-dual(constraint_by_name(model, "constr2[$i]")) for i=1:size(info2.tsupp, 2)]
MomMat2 = get_moment_matrix(moment2, info2.tsupp, info2.cql, info2.basis)

# retrieve Gram matrices
GramMat = Vector{Vector{Vector{Union{Float64,Matrix{Float64}}}}}(undef, info1.cql)
for i = 1:info1.cql
    GramMat[i] = Vector{Vector{Union{Float64,Matrix{Float64}}}}(undef, 1+length(info1.I[i])+length(info1.J[i]))
    for j = 1:1+length(info1.I[i])+length(info1.J[i])
        GramMat[i][j] = [value.(info1.gram[i][j][l]) for l = 1:info1.cl[i][j]]
    end
end

GramMat2 = Vector{Vector{Vector{Union{Float64,Matrix{Float64}}}}}(undef, info2.cql)
for i = 1:info2.cql
    GramMat2[i] = Vector{Vector{Union{Float64,Matrix{Float64}}}}(undef, 1+length(info2.I[i])+length(info2.J[i]))
    for j = 1:1+length(info2.I[i])+length(info2.J[i])
        GramMat2[i][j] = [value.(info2.gram[i][j][l]) for l = 1:info2.cl[i][j]]
    end
end

    
    

    

return objv, MomMat1, MomMat2,  model, [info1,info2] #GramMat, GramMat2
    
    
end








"""
    trace_penalty(m, sigma, d; weights=nothing, normalize=false)

Return the sum  ∑_{k=1}^d ∑_{i=1}^n (m_i^(2k) + sigma_i^(2k)).

Arguments
- `m`, `sigma`: vectors (length n) of variables or numbers (e.g. DynamicPolynomials).
- `d`::Int: highest even power order to include (powers 2,4,...,2d).

Keywords
- `weights`: optional vector of length d with nonnegative weights λ_k applied to order k.
- `normalize`::Bool: if true, divide the total by n to keep scale roughly invariant in n.

Notes
- Works for numeric vectors or polynomial variables (e.g., from TSSOS/DynamicPolynomials).
"""
function trace_penalty(m, sigma, d::Integer; weights=nothing, normalize::Bool=false)
    @assert length(m) == length(sigma) "m and sigma must have the same length"
    @assert d ≥ 1 "d must be ≥ 1"
    n = length(m)
    if weights === nothing
        λ = ones(d)
    else
        @assert length(weights) == d "weights must have length d"
        λ = weights
    end
    total = zero(m[1] + sigma[1])  # promotes to correct numeric/polynomial type
    for k in 1:d
        p = 2k
        w = λ[k]
        # sum of even powers for this order
        total += w * (sum(mi -> mi^p, m) + sum(si -> si^p, sigma))
    end
    return normalize ? total / n : total
end





function trace_like_expression(m::Vector{PolyVar{true}}, d::Int)
    vb = monomials(m, 1:d)  # exclude constant term (degree 0)
    expr = sum(mono^2 for mono in vb)
    return expr
end



function TV_SOS_model(n, d, m, sigma, S, samples; tr_reg=false,eps_tr=0.001)   

    
model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)

    
@polyvar x[1:n]
q, qc, qb = add_poly!(model, x, 2d)   #qc are coefficients of q
sigp, sigpc, sigpb = add_poly!(model, x, 2d)   #sigp are coefficients of sigma_+
sigm, sigmc, sigmb = add_poly!(model, x, 2d)   #sigp are coefficients of sigma_-




em_mom=compute_empirical_moments_from_basis(n, samples, qb, [x[i] for i=1:n])

obj_f=dot(em_mom,qc)-dot(em_mom,sigpc)         


sos1=1-q+sigp
model,info1 = add_psatz!(model, sos1, x, [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr1")

sos2=1+q+sigm
model,info2 = add_psatz!(model, sos2, x, [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr2")


gauss_moments=compute_theoretical_moments(n, qb, [x[i] for i=1:n], m, sigma)

if tr_reg
    sos3=sum((-qc[i]-sigmc[i])*gauss_moments[i] for i=1:length(sigmc)) + eps_tr*trace_like_expression(m,d)   # Trace of the moment matrix penalization kinda
else
    sos3=sum((-qc[i]-sigmc[i])*gauss_moments[i] for i=1:length(sigmc))
end
        
model,info3 = add_psatz!(model, sos3, [m;sigma], S, [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr3")

sos4=sigp
model,info4 = add_psatz!(model, sos4, x, [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr4")

sos5=sigm
model,info5 = add_psatz!(model, sos5, x, [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr5")


@objective(model, Max, obj_f)
optimize!(model)
objv = objective_value(model)
@show objv

# retrieve moment matrices
infos=[info1,info2,info3,info4,info5]

moment_list = Vector{Vector{Float64}}(undef, 5)
MomMat_list = Vector{Matrix{Float64}}(undef, 5)
    
for k in 1:5
    #info = getfield(Main, Symbol("info$(k)"))  # Only if info1, ..., info5 are global
    info=infos[k]
    moment_list[k] = [-dual(constraint_by_name(model, "constr$(k)[$i]")) for i in 1:size(info.tsupp, 2) ]
    MomMat_list[k] = get_moment_matrix(moment_list[k], info.tsupp, info.cql, info.basis)[1]
end



return objv, MomMat_list, MomMat_list[3],  model, [info1,info3] #GramMat, GramMat2
    
    
end





function multivariate_Gaussian_TV(n, d, m, sigma, S, samples, trace_penalization, vareps)   

    
model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)

    
@polyvar x[1:n]
q, qc, qb = add_poly!(model, x, 2d)   #qc are coefficients of q
sigp, sigpc, sigpb = add_poly!(model, x, 2d)   #sigp are coefficients of sigma_+
sigm, sigmc, sigmb = add_poly!(model, x, 2d)   #sigp are coefficients of sigma_-




em_mom=compute_empirical_moments_from_basis(n, samples, qb, [x[i] for i=1:n])

obj_f=-dot(em_mom,qc)-dot(em_mom,sigpc)         


sos1=1+q+sigp
model,info1 = add_psatz!(model, sos1, x, [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr1")

sos2=1-q+sigm
model,info2 = add_psatz!(model, sos2, x, [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr2")


gauss_moments=compute_theoretical_moments(n, qb, [x[i] for i=1:n], m, sigma)

if trace_penalization
    sos3=vareps*(trace_penalty(m, sigma, d)+trace_like_expression(m,d))+sum((qc[i]-sigmc[i])*gauss_moments[i] for i=1:length(sigmc))   # Trace of the moment matrix penalization kinda
else
    sos3=sum((qc[i]-sigmc[i])*gauss_moments[i] for i=1:length(sigmc))
end
        
model,info3 = add_psatz!(model, sos3, [m;sigma], S, [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr3")

sos4=sigp
model,info4 = add_psatz!(model, sos4, x, [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr4")

sos5=sigm
model,info5 = add_psatz!(model, sos5, x, [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr5")


@objective(model, Max, obj_f)
optimize!(model)
objv = objective_value(model)
@show objv

# retrieve moment matrices
infos=[info1,info2,info3,info4,info5]

moment_list = Vector{Vector{Float64}}(undef, 5)
MomMat_list = Vector{Matrix{Float64}}(undef, 5)
    
for k in 1:5
    #info = getfield(Main, Symbol("info$(k)"))  # Only if info1, ..., info5 are global
    info=infos[k]
    moment_list[k] = [-dual(constraint_by_name(model, "constr$(k)[$i]")) for i in 1:size(info.tsupp, 2) ]
    MomMat_list[k] = get_moment_matrix(moment_list[k], info.tsupp, info.cql, info.basis)[1]
end



return objv, MomMat_list, MomMat_list[3],  model, [info1,info3] #GramMat, GramMat2
    
    
end













function normalize_column(col)
    min_val = minimum(col)
    max_val = maximum(col)
    return (col .- min_val) ./ (max_val - min_val)
end



function extract_CF(Md, vd, s, n, r)
    subM = Md[1:s, 1:s]
    ic = sort(i_cols(subM, r))              
    subv = map(i -> vd[:, i], ic)

    C = Md[:, ic]
    hatH = Md[ic, ic]
    Cfact = cholesky(hatH)
    Cmat = Cfact.U                         

  

    Inm = UInt8.(Matrix(I, n, n))
    vdv = [Int.(vd[:, i]) for i in 1:size(vd, 2)]
    idxs = [indexin(map(m -> Int.(Inm[i, :]) + Int.(m), subv), vdv) for i in 1:n]

    X = Matrix{Float64}[]
    for i in 1:n
        idx_clean = sort(filter(!isnothing, idxs[i]))
        Cnext = Md[:, idx_clean]
        Abar = C \ Cnext
        Xi = Cmat * Abar * inv(Cmat)
        Xi = (Xi + Xi') / 2  # Symmetrize
        push!(X, Xi)
    end


    # Step 4: Random combination
    λ = rand(n)
    λ ./= norm(λ)
    N = sum(λ[i] * X[i] for i in 1:n)

    # Step 5: Schur decomposition
    T, Q, D = schur(Symmetric(N))

    # Step 6–7: Recover atoms
    #xsol = [map(i -> Q[:, j]' * X[i] * Q[:, j], 1:n) for j in 1:r]
    xsol = [ [ Q[:,j]' * Xi * Q[:,j] for Xi in X ] for j in 1:r ]

    #New way
    #xsol = map(j -> [Q[j,:]'*X1*Q[j,:], Q[j,:]'*X2*Q[j,:]],1:r);
    #xsol = [map(i -> Q[j,:]' * X[i] * Q[j,:], 1:n) for j in 1:r]
    
    return xsol
end

function extract_marginal_en_m(mat, basis)
    col_indices = findall(j -> basis[3, j] == 0 && basis[4, j] == 0, 1:size(basis, 2))
    return [col_indices, mat[col_indices,col_indices]]
end

function extract_marginal_en_mm(mat, basis, m_dim::Int)
    # Discard any monomial with nonzero degree in non-m variables (rows m_dim+1:end)
    col_indices = findall(j -> all(basis[i, j] == 0 for i in m_dim+1:size(basis, 1)), 1:size(basis, 2))
    return col_indices, mat[col_indices, col_indices]
end

function extract_marginal_sigma(mat, basis, m_dim::Int)
    # Keep columns where all degrees in the first m_dim rows are zero
    col_indices = findall(j -> all(basis[i, j] == 0 for i in 1:m_dim), 1:size(basis, 2))
    return col_indices, mat[col_indices, col_indices]
end


using LinearAlgebra

# Unwrap a Matrix from either a Matrix or a container holding a Matrix
function _first_matrix(x)
    if x isa AbstractMatrix
        return Matrix(x)
    elseif x isa AbstractVector || x isa Tuple
        for e in x
            e isa AbstractMatrix && return Matrix(e)
        end
    end
    error("Expected a Matrix or a container holding a Matrix, got $(typeof(x)).")
end

_eigvals_sorted(M) = sort(issymmetric(M) ? eigen(Symmetric(M)).values : eigen(M).values; rev=true)

function analyse_relaxations_W2orTV(res, d::Integer, mdim::Integer, energy_tol)
    # FIX: unwrap from res[d][3], not res[d][3][1]
    M_d   = _first_matrix(res[d][3])
    M_dm1 = _first_matrix(res[d-1][3])

    basis_d   = res[d][end][2].basis[1][1]
    basis_dm1 = res[d-1][end][2].basis[1][1]

    bd,   M_d_mm   = extract_marginal_en_mm(M_d,   basis_d,   mdim)
    bd1,  M_dm1_mm = extract_marginal_en_mm(M_dm1, basis_dm1, mdim)

    # Use the overlapping size to be extra-robust
    sz = min(size(M_d_mm, 1), size(M_dm1_mm, 1))
    sub_M_d_mm = M_d_mm[1:sz, 1:sz]

    ev_d     = _eigvals_sorted(M_d_mm)
    ev_d_sub = _eigvals_sorted(sub_M_d_mm)

    basis_KF = basis_d[1:mdim, bd]

    println("Eigenvalues of the moment matrix of order d = $d (top 10):")
    println(ev_d[1:min(10, length(ev_d))])
    println("Rank at d = ", rank_by_energy(M_d_mm; energy_tol=float(energy_tol)))

    println("Eigenvalues of the sub-matrix of order d-1 = $(d-1) (top 10):")
    println(ev_d_sub[1:min(10, length(ev_d_sub))])
    println("Rank at d-1 = ", rank_by_energy(sub_M_d_mm; energy_tol=float(energy_tol)))

    return M_d_mm, sub_M_d_mm, M_dm1_mm, basis_KF
end



function analyse_relaxations(res, d, mdim, energy_tol)

    bd, mmd = extract_marginal_en_mm(res[d][3][1],res[d][end][2].basis[1][1],mdim)
    bdminus1, mmdminus1 = extract_marginal_en_mm(res[d-1][3][1],res[d-1][end][2].basis[1][1],mdim)

    sub_mmd = mmd[1:size(mmdminus1,1),1:size(mmdminus1,1)]

    ev_mmd = eigen(mmd).values

    ev_sub_mmd = eigen(sub_mmd).values

    basis_KF = res[d][end][2].basis[1][1][1:n, bd] 
    
    

    

    println("Eigenvalues of the moment matrix of order  d = ", d, ":")
    println(sort(ev_mmd, rev=true)[1:min(10,length(ev_mmd))])
    println("Rank at d = ", rank_by_energy(mmd; energy_tol=energy_tol))
    println("Eigenvalues of the sub-matrix of order d-1 = ", d-1, ":")
    println(sort(ev_sub_mmd, rev=true)[1:min(10,length(ev_sub_mmd))])
    println("Rank at d = ", rank_by_energy(sub_mmd; energy_tol=energy_tol))

    return mmd, sub_mmd, mmdminus1, basis_KF

end 

function analyse_relaxations_TV(res, d, mdim)
    
    bd, mmd = extract_marginal_en_mm(res[d][3],res[d][end][2].basis[1][1],mdim)
    bdminus1, mmdminus1 = extract_marginal_en_mm(res[d-1][3],res[d-1][end][2].basis[1][1],mdim)

    sub_mmd = mmd[1:size(mmdminus1,1),1:size(mmdminus1,1)]

    ev_mmd = eigen(mmd).values

    ev_sub_mmd = eigen(sub_mmd).values

    basis_KF = res[d][end][2].basis[1][1][1:n, bd] 
    

    

    println("Eigenvalues of the moment matrix of order ", d, ":")
    println(sort(ev_mmd, rev=true)[1:min(10,length(ev_mmd))])
    println()
    println("Eigenvalues of the sub-matrix of order ", d-1, ":")
    println(sort(ev_sub_mmd, rev=true)[1:min(10,length(ev_sub_mmd))])

    return mmd, sub_mmd, mmdminus1, basis_KF

end 


function analyse_relaxations_sigma(res, d, mdim)
    bd, mmd = extract_marginal_sigma(res[d][3][1],res[d][end][2].basis[1][1],mdim)
    bdminus1, mmdminus1 = extract_marginal_sigma(res[d-1][3][1],res[d-1][end][2].basis[1][1],mdim)

    sub_mmd = mmd[1:size(mmdminus1,1),1:size(mmdminus1,1)]

    ev_mmd = eigen(mmd).values

    ev_sub_mmd = eigen(sub_mmd).values

    basis_KF = res[d][end][2].basis[1][1][(n+1):end, bd]

    

    println("Eigenvalues of the marginal moment matrix of order ", d, ":")
    println(sort(ev_mmd, rev=true)[1:min(10,length(ev_mmd))])
    println()
    println("Eigenvalues of the marginal sub-matrix of order ", d-1, ":")
    println(sort(ev_sub_mmd, rev=true)[1:min(10,length(ev_sub_mmd))])

    return mmd, sub_mmd, mmdminus1, basis_KF

end 




_spectrum(M) = sort!(max.(0.0, abs.(eigvals(Symmetric(Matrix(M))))), rev=true)

"""
    rank_by_energy(M; energy_tol=1e-3)

Smallest r s.t. sum_{i=1}^r λ_i ≥ (1 - energy_tol) * sum(λ).
"""
function rank_by_energy(M; energy_tol=1e-3)
    λ = _spectrum(M)
    tot = sum(λ)
    return tot == 0 ? 0 : searchsortedfirst(cumsum(λ), (1 - energy_tol) * tot)
end





#######################################################################################################################

#################################################         UNIVARIATE CASE - Gaussian

#######################################################################################################################



####### Projections on axes - i,e. Marginal Optimal Transports

function double_fact(n)
    n ≤ 1 && return 1
    return prod(n:-2:1)
end

function gaussian_moments_univariate_up_to(order::Int, m, sigma)
    """
    Computes symbolic univariate Gaussian moments E[X^k] for k = 0 to `order`.

    Arguments:
    - order: maximum degree of the moment
    - m: mean of the distribution (symbolic or numeric)
    - sigma: standard deviation (symbolic or numeric)

    Returns:
    - Vector of symbolic expressions [E[X^0], E[X^1], ..., E[X^order]]
    """
    moments = Vector{typeof(m + sigma)}(undef, order + 1)

    for k in 0:order
        moment = zero(m)
        for j in 0:floor(Int, k / 2)
            coeff = binomial(k, 2j) * double_fact(2j - 1)
            term = coeff * sigma^(2j) * m^(k - 2j)
            moment += term
        end
        moments[k + 1] = moment  # index starts from 1 in Julia
    end

    return moments
end

function compute_empirical_moments_univariate(samples::Vector{<:Real}, order)
    """
    Computes empirical moments for a univariate dataset based on given monomials.

    Arguments:
    - samples: Vector of real-valued samples from the distribution
    - monomials: Vector of monomials like [1, x, x^2, x^3, ...]

    Returns:
    - Vector of empirical moments in the order of `monomials`
    """
    N = length(samples)
    return [sum(x^i for x in samples) / N for i=0:order]
end




function univariate_SOS_model_Gaussian_W2(d, m, sigma, S, samples, trace_penalization, vareps)

    
model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)

    
@polyvar x
@polyvar y
q, qc, qb = add_poly!(model, x, 2d)
g, gc, gb = add_poly!(model, y, 2d)

em_mom=compute_empirical_moments_univariate(samples, 2d)

obj_f=dot(em_mom,qc)


sos1=dot(x-y,x-y)-dot(qc,qb)-dot(gc,gb)
model,info1 = add_psatz!(model, sos1, [x;y], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr1")


gauss_moments=gaussian_moments_univariate_up_to(2d, m, sigma)
    

if trace_penalization
    sos2=vareps*(1+sum(m^(2k) for k in 1:d)+sum(sigma^(2k) for k in 1:d))-sum(gc[i]*gauss_moments[i] for i=1:length(gc))   
else
    sos2=-sum(gc[i]*gauss_moments[i] for i=1:length(gc))
end
        
model,info2 = add_psatz!(model, sos2, [m;sigma], S, [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr2")

@objective(model, Max, obj_f)
optimize!(model)
objv = objective_value(model)
@show objv

    
# retrieve moment matrices
moment1 = [-dual(constraint_by_name(model, "constr1[$i]")) for i=1:size(info1.tsupp, 2)]
MomMat1 = get_moment_matrix(moment1, info1.tsupp, info1.cql, info1.basis)

moment2 = [-dual(constraint_by_name(model, "constr2[$i]")) for i=1:size(info2.tsupp, 2)]
MomMat2 = get_moment_matrix(moment2, info2.tsupp, info2.cql, info2.basis)


return objv, MomMat1, MomMat2, model, [info1,info2]
    
    
    
end


function relative_rank(values,threshold = 1e-3)

    significant = count(x -> x / maximum(values) > threshold, values)

    return significant
end


function scale_to_minus1_1(col)
    min_val = minimum(col)
    max_val = maximum(col)
    #return @. 2 * (col - min_val) / (max_val - min_val) - 1
    return @. (col - min_val) / (max_val - min_val) 
end












function extract(Md,vd,s,n)

subM=Md[1:s,1:s];
ic = sort(i_cols(subM,r));
subv = map(i->vd[:,i],ic);
C = Md[:,ic];
hatH = Md[ic,ic];
G = cholesky(hatH).U;

In=UInt8.(Matrix(I,n,n));

idx1 = map(m -> In[1,:]+m ,subv);
idx2 = map(m -> In[2,:]+m ,subv);

vdv = [vd[:,i] for i in 1:size(vd,2)];

# indexes of monomials obtained after multiplication by x[1] and x[2]

Cidx1 = indexin(idx1,vdv);
Cidx2 = indexin(idx2,vdv);

C1 = Md[:,sort(Cidx1)];
C2 = Md[:,sort(Cidx2)];

Ax1bar = C \ C1;
Ax2bar = C \ C2;

# X1 and X2 are the matrices encoding the multiplication operators

X1 = G * Ax1bar * inv(G);
X2 = G * Ax2bar * inv(G);

X1=(X1+X1')/2;
X2=(X2+X2')/2;

# Now we extract the atoms of the Dirac measure

rn = rand(n);
rn /= norm(rn);
A = rn[1]*X1+rn[2]*X2;

T, Q, D = schur(Symmetric(A));

# each element of xsol is an extracted solution
xsol = map(j -> [Q[j,:]'*X1*Q[:,j], Q[j,:]'*X2*Q[:,j]],1:r);

    #New way
#xsol = map(j -> [Q[j,:]'*X1*Q[j,:], Q[j,:]'*X2*Q[j,:]],1:r);

#xsol = map(j -> [Q[j,:]'*X1*Q[j,:], Q[j,:]'*X2*Q[j,:]],1:r);

return xsol
end


function i_cols(A, r)
  QR = qr(A, Val(true))
    return QR.p[1:r]
end






#################################### MIXTRUES OF UNIVARIATE EXPONENTIALS ##################################
############################################################################################################


function exponential_moments_univariate_up_to(order::Int, mu)
    """
    Computes symbolic or numeric univariate Exponential moments E[X^k] for k = 0 to `order`,
    under reparametrization by the mean mu = 1 / lambda.

    Arguments:
    - order: maximum degree of the moment
    - mu: mean of the Exponential distribution (symbolic or numeric)

    Returns:
    - Vector of expressions [E[X^0], E[X^1], ..., E[X^order]]
    """
    moments = Vector{Any}(undef, order + 1)

    for k in 0:order
        moment = factorial(k) * mu^k
        moments[k + 1] = moment
    end

    return moments
end


function univariate_SOS_model_EXP(d, mu, S, samples, trace_penalization=false)

    
model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)

    
@polyvar x
@polyvar y
q, qc, qb = add_poly!(model, x, 2d)
g, gc, gb = add_poly!(model, y, 2d)

em_mom=compute_empirical_moments_univariate(samples, 2d)

obj_f=dot(em_mom,qc)


sos1=dot(x-y,x-y)-dot(qc,qb)-dot(gc,gb)
model,info1 = add_psatz!(model, sos1, [x;y], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr1")


expo_moments=exponential_moments_univariate_up_to(2d, mu)

if trace_penalization
    sos2=sum(gc[i]*expo_moments[i] for i=1:length(gc)) + 0.001*sum(mu^(2i) for i in 1:d)
else
    sos2=sum(gc[i]*expo_moments[i] for i=1:length(gc)) 
end
        
model,info2 = add_psatz!(model, sos2, [mu], [S], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr2")

@objective(model, Max, obj_f)
optimize!(model)
objv = objective_value(model)
@show objv

    
# retrieve moment matrices
moment1 = [-dual(constraint_by_name(model, "constr1[$i]")) for i=1:size(info1.tsupp, 2)]
MomMat1 = get_moment_matrix(moment1, info1.tsupp, info1.cql, info1.basis)

moment2 = [-dual(constraint_by_name(model, "constr2[$i]")) for i=1:size(info2.tsupp, 2)]
MomMat2 = get_moment_matrix(moment2, info2.tsupp, info2.cql, info2.basis)


return objv, MomMat1, MomMat2, model, [info1,info2]
    
    
    
end







################## Mixture of univariate Poisson distributions ##########################

function empirical_moments_univariate_poisson_up_to(order::Int, data::AbstractVector{<:Integer})
    """
    Computes empirical moments E[X^k] for k = 0 to `order`
    from a sample of Poisson-distributed counts.

    Arguments:
    - order: maximum degree of the moment
    - data: vector of observed counts (Int)

    Returns:
    - Vector of Float64 values [M_0, M_1, ..., M_order]
      where M_k = (1/n) * sum_i data[i]^k
    """
    n = length(data)
    @assert n > 0 "Data vector cannot be empty"

    moments = Vector{Float64}(undef, order + 1)
    for k in 0:order
        moments[k+1] = mean(x -> x^k, data)
    end

    return moments
end


# Stirling numbers of the 2nd kind up to `order`
function _stirling2_table(order::Int)
    S = zeros(Int, order+1, order+1)   # S[n+1,k+1] ≡ S(n,k)
    S[1,1] = 1                          # S(0,0)=1
    for n in 1:order
        for k in 1:n
            # S(n,k) = S(n-1,k-1) + k*S(n-1,k)
            S[n+1, k+1] = S[n, k] + k*S[n, k+1]
        end
    end
    return S
end

"""
    poisson_raw_moments_up_to(order, λ)

Return the vector [E[X^0], …, E[X^order]] for X ~ Poisson(λ).
Works when λ is numeric or a symbolic polynomial (e.g. created with `@polyvar`).
"""
function poisson_raw_moments_up_to(order::Int, λ)
    @assert order >= 0
    S = _stirling2_table(order)

    moments = Vector{Any}(undef, order+1)
    moments[1] = λ^0                    # E[X^0] = 1, with the right coefficient type
    for k in 1:order
        acc = zero(λ)                   # keeps numeric/symbolic type
        for j in 0:k
            acc += S[k+1, j+1] * (λ^j)
        end
        moments[k+1] = acc
    end
    return moments
end


function univariate_SOS_model_Poisson(d, lambda, S, samples, trace_penalization=false)

    
model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)

    
@polyvar x
@polyvar y
q, qc, qb = add_poly!(model, x, 2d)
g, gc, gb = add_poly!(model, y, 2d)

em_mom=compute_empirical_moments_univariate(samples, 2d)

obj_f=dot(em_mom,qc)


sos1=dot(x-y,x-y)-dot(qc,qb)-dot(gc,gb)
model,info1 = add_psatz!(model, sos1, [x;y], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr1")


poisson_moments=poisson_raw_moments_up_to(2*d, lambda)

sos2 = sum(gc[i] * poisson_moments[i] for i in 1:length(gc))
if trace_penalization
    sos2 += 1e-2 * sum(lambda^(2*i) for i in 1:d)
end
        
model,info2 = add_psatz!(model, sos2, [lambda], [S], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr2")

@objective(model, Max, obj_f)
optimize!(model)
objv = objective_value(model)
@show objv

    
# retrieve moment matrices
moment1 = [-dual(constraint_by_name(model, "constr1[$i]")) for i=1:size(info1.tsupp, 2)]
MomMat1 = get_moment_matrix(moment1, info1.tsupp, info1.cql, info1.basis)

moment2 = [-dual(constraint_by_name(model, "constr2[$i]")) for i=1:size(info2.tsupp, 2)]
MomMat2 = get_moment_matrix(moment2, info2.tsupp, info2.cql, info2.basis)


return objv, MomMat1, MomMat2, model, [info1,info2]
    
    
    
end






function univariate_TV_SOS_model_Poisson(d, lambda, S, samples; tr_reg=false,eps_tr=0.001)   

    
model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)

    
@polyvar x[1:n]
q, qc, qb = add_poly!(model, x, 2d)   #qc are coefficients of q
sigp, sigpc, sigpb = add_poly!(model, x, 2d)   #sigp are coefficients of sigma_+
sigm, sigmc, sigmb = add_poly!(model, x, 2d)   #sigp are coefficients of sigma_-




em_mom=compute_empirical_moments_univariate(samples, 2d)

obj_f=dot(em_mom,qc)-dot(em_mom,sigpc)         


sos1=1-q+sigp
model,info1 = add_psatz!(model, sos1, x, [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr1")

sos2=1+q+sigm
model,info2 = add_psatz!(model, sos2, x, [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr2")


gauss_moments=compute_theoretical_moments(n, qb, [x[i] for i=1:n], m, sigma)

if tr_reg
    sos3=sum((-qc[i]-sigmc[i])*gauss_moments[i] for i=1:length(sigmc)) + eps_tr*trace_like_expression(m,d)   # Trace of the moment matrix penalization kinda
else
    sos3=sum((-qc[i]-sigmc[i])*gauss_moments[i] for i=1:length(sigmc))
end
        
model,info3 = add_psatz!(model, sos3, [m;sigma], S, [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr3")

sos4=sigp
model,info4 = add_psatz!(model, sos4, x, [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr4")

sos5=sigm
model,info5 = add_psatz!(model, sos5, x, [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr5")


@objective(model, Max, obj_f)
optimize!(model)
objv = objective_value(model)
@show objv

# retrieve moment matrices
infos=[info1,info2,info3,info4,info5]

moment_list = Vector{Vector{Float64}}(undef, 5)
MomMat_list = Vector{Matrix{Float64}}(undef, 5)
    
for k in 1:5
    #info = getfield(Main, Symbol("info$(k)"))  # Only if info1, ..., info5 are global
    info=infos[k]
    moment_list[k] = [-dual(constraint_by_name(model, "constr$(k)[$i]")) for i in 1:size(info.tsupp, 2) ]
    MomMat_list[k] = get_moment_matrix(moment_list[k], info.tsupp, info.cql, info.basis)[1]
end



return objv, MomMat_list, MomMat_list[3],  model, [info1,info3] #GramMat, GramMat2
    
    
end







###############################################




# --- Touchard (raw) moments of Poisson(λ), works for numeric or symbolic λ
function _stirling2_table(order::Int)
    S = zeros(Int, order+1, order+1)   # S[n+1,k+1] = S(n,k)
    S[1,1] = 1
    for n in 1:order, k in 1:n
        S[n+1,k+1] = S[n,k] + k*S[n,k+1]   # S(n,k)=S(n-1,k-1)+k*S(n-1,k)
    end
    return S
end

function poisson_raw_moments_up_to(order::Int, λ)
    S = _stirling2_table(order)
    moms = Vector{Any}(undef, order+1)
    moms[1] = one(λ)     # m0 = 1
    for k in 1:order
        acc = zero(λ)
        for j in 0:k
            acc += S[k+1, j+1] * (λ^j)
        end
        moms[k+1] = acc
    end
    return moms
end




"""
univariate_TV_SOS_model_Poisson(d, λ, S_ineqs, samples; δ=0.0, ε=0.0)

Implements the TV-regularized SOS dual (univariate version of (23a–23e))
for mixtures of Poisson laws mixed over the rate λ ∈ Θ.

Arguments:
- d::Int : relaxation degree
- λ      : either a DynamicPolynomials variable (if optimizing over λ) or a numeric scalar (fixed)
- S_ineqs::Vector{Polynomial} : inequality polynomials r_j(λ) ≥ 0 describing Θ
- samples::Vector{Int} : observed counts
Keyword:
- δ::Real : total-variation penalty/budget coefficient in (23a)
- ε::Real : regularization weight for R(λ) inside (23d)

Returns:
  objv, MomMats, model, infos
"""
function univariate_TV_SOS_model_Poissonn(d::Int, lambda, S_ineqs, samples;
                                         delta::Real=0.0, epsy::Real=0.0)

    model = Model(optimizer_with_attributes(Mosek.Optimizer))
    set_optimizer_attribute(model, MOI.Silent(), true)

    # ----- polynomials in x (data space)
    @polyvar x
    # q^+, q^-, σ^+, σ^-  with degrees ≤ 2d (q’s) and ≤ 2d (we enforce SOS via add_psatz!)
    qplus,  qc_p, qb_p = add_poly!(model, x, 2*d)   # coefficients variables qc_p
    qminus, qc_m, qb_m = add_poly!(model, x, 2*d)
    sigp,   sigpc, _   = add_poly!(model, x, 2*d)
    sigm,   sigmc, _   = add_poly!(model, x, 2*d)

    # empirical moments μ^N_α for α = 0..2d
    mu_N = compute_empirical_moments_univariate(samples, 2*d)

    # (23a) objective: sum_α (q^+_α - σ^+_α) μ_α^N  - δ * sum_α (q^+_α + q^-_α)
    obj = dot(mu_N , qc_m) - dot(mu_N , qc_p) - dot(mu_N , sigpc) - delta*(sum(qc_p) + sum(qc_m))

    # (23b)  1 + q^+ - q^- + σ^+ ∈ Σ_d[x]
    poly_b = 1 + qplus - qminus + sigp
    model, info1 = add_psatz!(model, poly_b, [x], [], [], d,
                              QUIET=true, CS=false, TS=false, Groebnerbasis=false, constrs="constr1")

    # (23c)  1 - q^+ + q^- + σ^- ∈ Σ_d[x]
    poly_c = 1 - qplus + qminus + sigm
    model, info2 = add_psatz!(model, poly_c, [x], [], [], d,
                              QUIET=true, CS=false, TS=false, Groebnerbasis=false, constrs="constr2")

    # (23d)  ε R(λ) + ∑_α (q^+_α - q^-_α - σ^-_α) p_α(λ) ∈ Q_d(r)
    poisson_moms=poisson_raw_moments_up_to(2*d, lambda)
   # Touchard polynomials in λ
    poisson_moments=[poisson_moms[i]/maximum(coefficients(poisson_moms[i])) for i=1:length(poisson_moms)]

    # Simple polynomial regularizer R(λ) used in (23d)
    trace_like_R(lambda, d) = sum(lambda^(2*i) for i in 1:d)
    
    poly_d = epsy*trace_like_R(lambda, d) + sum((qc_p[i] - qc_m[i] - sigmc[i]) * poisson_moments[i] for i in 1:length(poisson_moments))
    #poly_d=poly_d/maximum(coefficients(poly_d))


    model, info3 = add_psatz!(model, poly_d, [lambda], [S_ineqs], [], d,
                                  QUIET=true, CS=false, TS=false, Groebnerbasis=false, constrs="constr3")

    # Enforce σ^± ∈ Σ_d[x] explicitly (keeps them SOS on their own)
    model, info4 = add_psatz!(model, sigp, [x], [], [], d,
                              QUIET=true, CS=false, TS=false, Groebnerbasis=false, constrs="constr4")
    model, info5 = add_psatz!(model, sigm, [x], [], [], d,
                              QUIET=true, CS=false, TS=false, Groebnerbasis=false, constrs="constr5")

     # (23e) coefficients of q^± are nonnegative
    @constraint(model, qc_p .>= 0)
    @constraint(model, qc_m .>= 0)

    @objective(model, Max, obj)
    optimize!(model)
    objv = objective_value(model)

    

    # retrieve moment matrices
    infos=[info1,info2,info3,info4,info5]
    
    moment_list = Vector{Vector{Float64}}(undef, 5)
    MomMat_list = Vector{Matrix{Float64}}(undef, 5)
        
    for k in 1:5
        #info = getfield(Main, Symbol("info$(k)"))  # Only if info1, ..., info5 are global
        info=infos[k]
        moment_list[k] = [-dual(constraint_by_name(model, "constr$(k)[$i]")) for i in 1:size(info.tsupp, 2)]
        MomMat_list[k] = get_moment_matrix(moment_list[k], info.tsupp, info.cql, info.basis)[1]
    end

    return objv, MomMat_list, MomMat_list[3],  model, [info1,info3] #GramMat, GramMat2

    #return objv, MomMats, model, infos
end











