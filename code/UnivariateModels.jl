using JuMP
using MosekTools
using DynamicPolynomials
using MultivariatePolynomials
using TSSOS
using LinearAlgebra, Random, Plots, Distributions, IterTools, Combinatorics




function double_fact(n)
    n ≤ 1 && return 1
    return prod(n:-2:1)
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





############################################################  GAUSSIAN #################################################################

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



function univariate_SOS_model_Gaussian_W2(d, m, sigma, S, samples, trace_penalization, vareps)

    
model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)
@polyvar x
@polyvar y
q, qc, qb = add_poly!(model, x, 2d)
g, gc, gb = add_poly!(model, y, 2d)

em_mom=compute_empirical_moments_univariate(samples, 2d)

obj_f=dot(em_mom,qc)
    
sos1=dot(x-y,x-y)-dot(qc,qb)+dot(gc,gb)
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



function univariate_SOS_model_Gaussian_TV(d, m, sigma, S, samples, trace_penalization, vareps)   

model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)
    
@polyvar x

q, qc, qb = add_poly!(model, x, 2d)   #qc are coefficients of q
sigp, sigpc, sigpb = add_poly!(model, x, 2d)   #sigp are coefficients of sigma_+
sigm, sigmc, sigmb = add_poly!(model, x, 2d)   #sigp are coefficients of sigma_-

em_mom=compute_empirical_moments_univariate(samples, 2d)

obj_f=dot(em_mom,-qc-sigpc)       
    
sos1=1+q+sigp
model,info1 = add_psatz!(model, sos1, [x], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr1")

sos2=1-q+sigm
model,info2 = add_psatz!(model, sos2, [x], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr2")

gauss_moments=gaussian_moments_univariate_up_to(2d, m, sigma)

if trace_penalization
    sos3 = vareps*(1+sum(m^(2k) +sigma^(2k) for k in 1:d))+sum((qc[i]-sigmc[i])*gauss_moments[i] for i=1:length(qc))   
else
    sos3= sum((qc[i]-sigmc[i])*gauss_moments[i] for i=1:length(qc)) 
end   
model,info3 = add_psatz!(model, sos3, [m;sigma], S, [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr3")

sos4=sigp
model,info4 = add_psatz!(model, sos4, [x], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr4")

sos5=sigm
model,info5 = add_psatz!(model, sos5, [x], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr5")


@objective(model, Max, obj_f)
optimize!(model)
objv = objective_value(model)
@show objv

# retrieve moment matrices
infos=[info1,info2,info3,info4,info5]

#moment_list = Vector{Vector{Float64}}(undef, 5)
#MomMat_list = Vector{Matrix{Float64}}(undef, 5)
    
#for k in 1:5
    #info = getfield(Main, Symbol("info$(k)"))  # Only if info1, ..., info5 are global
    #info=infos[k]
    #moment_list[k] = [-dual(constraint_by_name(model, "constr$(k)[$i]")) for i in 1:size(info.tsupp, 2) ]
    #MomMat_list[k] = get_moment_matrix(moment_list[k], info.tsupp, info.cql, info.basis)[1]
#end

    # retrieve moment matrices
moment1 = [-dual(constraint_by_name(model, "constr1[$i]")) for i=1:size(info1.tsupp, 2)]
MomMat1 = get_moment_matrix(moment1, info1.tsupp, info1.cql, info1.basis)
    
moment2 = [-dual(constraint_by_name(model, "constr2[$i]")) for i=1:size(info2.tsupp, 2)]
MomMat2 = get_moment_matrix(moment2, info2.tsupp, info2.cql, info2.basis)
    
moment3 = [-dual(constraint_by_name(model, "constr3[$i]")) for i=1:size(info3.tsupp, 2)]
MomMat3 = get_moment_matrix(moment3, info3.tsupp, info3.cql, info3.basis)
    
moment4 = [-dual(constraint_by_name(model, "constr4[$i]")) for i=1:size(info4.tsupp, 2)]
MomMat4 = get_moment_matrix(moment4, info4.tsupp, info4.cql, info4.basis)
    
moment5 = [-dual(constraint_by_name(model, "constr5[$i]")) for i=1:size(info5.tsupp, 2)]
MomMat5 = get_moment_matrix(moment5, info5.tsupp, info5.cql, info5.basis)

#return objv , MomMat_list, MomMat_list[3],  model, [info1,info3] #GramMat, GramMat2
return objv , [MomMat1,MomMat2,MomMat3,MomMat4,MomMat5], MomMat3,  model, [info1,info3] 
    
end



####################################################################  POISSON #########################################################################




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
function poisson_moments_univariate_up_to(order::Int, lambda)
    @assert order >= 0
    S = _stirling2_table(order)

    moments = Vector{Any}(undef, order+1)
    moments[1] = lambda^0                    # E[X^0] = 1, with the right coefficient type
    for k in 1:order
        acc = zero(lambda)                   # keeps numeric/symbolic type
        for j in 0:k
            acc += S[k+1, j+1] * (lambda^j)
        end
        moments[k+1] = acc
    end
    return moments
end


function univariate_SOS_model_Poisson_W2(d, lambda, S, samples, trace_penalization, vareps)

    
model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)
@polyvar x
@polyvar y
q, qc, qb = add_poly!(model, x, 2d)
g, gc, gb = add_poly!(model, y, 2d)

em_mom=compute_empirical_moments_univariate(samples, 2d)

obj_f=dot(em_mom,qc)
    
sos1=dot(x-y,x-y)-dot(qc,qb)+dot(gc,gb)
model,info1 = add_psatz!(model, sos1, [x;y], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr1")

poisson_moments=poisson_moments_univariate_up_to(2d, lambda)

if trace_penalization
    sos2=vareps*(1+sum(lambda^(2k) for k in 1:d))-sum(gc[i]*poisson_moments[i] for i=1:length(gc))   
else
    sos2=-sum(gc[i]*poisson_moments[i] for i=1:length(gc))
end
        
model,info2 = add_psatz!(model, sos2, [lambda], S, [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr2")

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




function univariate_SOS_model_Poisson_TV(d, lambda, S, samples, trace_penalization, vareps)   

model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)
    
@polyvar x

q, qc, qb = add_poly!(model, x, 2d)   #qc are coefficients of q
sigp, sigpc, sigpb = add_poly!(model, x, 2d)   #sigp are coefficients of sigma_+
sigm, sigmc, sigmb = add_poly!(model, x, 2d)   #sigp are coefficients of sigma_-

em_mom=compute_empirical_moments_univariate(samples, 2d)

obj_f=dot(em_mom,-qc-sigpc)       
    
sos1=1+q+sigp
model,info1 = add_psatz!(model, sos1, [x], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr1")

sos2=1-q+sigm
model,info2 = add_psatz!(model, sos2, [x], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr2")

poisson_moments=poisson_moments_univariate_up_to(2d, lambda)

if trace_penalization
    sos3 = vareps*(1+sum(lambda^(2k) for k in 1:d))+sum((qc[i]-sigmc[i])*poisson_moments[i] for i=1:length(qc))   
else
    sos3= sum((qc[i]-sigmc[i])*poisson_moments[i] for i=1:length(qc)) 
end   
model,info3 = add_psatz!(model, sos3, [lambda], S, [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr3")

sos4=sigp
model,info4 = add_psatz!(model, sos4, [x], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr4")

sos5=sigm
model,info5 = add_psatz!(model, sos5, [x], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr5")


@objective(model, Max, obj_f)
optimize!(model)
objv = objective_value(model)
@show objv

# retrieve moment matrices
infos=[info1,info2,info3,info4,info5]

#moment_list = Vector{Vector{Float64}}(undef, 5)
#MomMat_list = Vector{Matrix{Float64}}(undef, 5)
    
#for k in 1:5
    #info = getfield(Main, Symbol("info$(k)"))  # Only if info1, ..., info5 are global
    #info=infos[k]
    #moment_list[k] = [-dual(constraint_by_name(model, "constr$(k)[$i]")) for i in 1:size(info.tsupp, 2) ]
    #MomMat_list[k] = get_moment_matrix(moment_list[k], info.tsupp, info.cql, info.basis)[1]
#end

    # retrieve moment matrices
moment1 = [-dual(constraint_by_name(model, "constr1[$i]")) for i=1:size(info1.tsupp, 2)]
MomMat1 = get_moment_matrix(moment1, info1.tsupp, info1.cql, info1.basis)
    
moment2 = [-dual(constraint_by_name(model, "constr2[$i]")) for i=1:size(info2.tsupp, 2)]
MomMat2 = get_moment_matrix(moment2, info2.tsupp, info2.cql, info2.basis)
    
moment3 = [-dual(constraint_by_name(model, "constr3[$i]")) for i=1:size(info3.tsupp, 2)]
MomMat3 = get_moment_matrix(moment3, info3.tsupp, info3.cql, info3.basis)
    
moment4 = [-dual(constraint_by_name(model, "constr4[$i]")) for i=1:size(info4.tsupp, 2)]
MomMat4 = get_moment_matrix(moment4, info4.tsupp, info4.cql, info4.basis)
    
moment5 = [-dual(constraint_by_name(model, "constr5[$i]")) for i=1:size(info5.tsupp, 2)]
MomMat5 = get_moment_matrix(moment5, info5.tsupp, info5.cql, info5.basis)

#return objv , MomMat_list, MomMat_list[3],  model, [info1,info3] #GramMat, GramMat2
return objv , [MomMat1,MomMat2,MomMat3,MomMat4,MomMat5], MomMat3,  model, [info1,info3] 
    
end




##########################################



#################################### MIXTRUES OF UNIVARIATE EXPONENTIALS ##################################
############################################################################################################


"""
    exponential_moments_univariate_up_to(order, μ)

Raw moments [E[X^0], …, E[X^order]] for X ~ Exp(μ).
E[X^k] = k! * μ^k.
"""
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




function univariate_SOS_model_Exponential_W2(d, mu, S, samples, trace_penalization, vareps)

    
model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)
@polyvar x
@polyvar y
q, qc, qb = add_poly!(model, x, 2d)
g, gc, gb = add_poly!(model, y, 2d)

em_mom=compute_empirical_moments_univariate(samples, 2d)

obj_f=dot(em_mom,qc)
    
sos1=dot(x-y,x-y)-dot(qc,qb)+dot(gc,gb)
model,info1 = add_psatz!(model, sos1, [x;y], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr1")

exponential_moments=exponential_moments_univariate_up_to(2d, mu)

if trace_penalization
    sos2=vareps*(1+sum(mu^(2k) for k in 1:d))-sum(gc[i]*exponential_moments[i] for i=1:length(gc))   
else
    sos2=-sum(gc[i]*exponential_moments[i] for i=1:length(gc))
end
        
model,info2 = add_psatz!(model, sos2, [mu], S, [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr2")

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



function univariate_SOS_model_Exponential_TV(d, mu, S, samples, trace_penalization, vareps)   

model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)
    
@polyvar x

q, qc, qb = add_poly!(model, x, 2d)   #qc are coefficients of q
sigp, sigpc, sigpb = add_poly!(model, x, 2d)   #sigp are coefficients of sigma_+
sigm, sigmc, sigmb = add_poly!(model, x, 2d)   #sigp are coefficients of sigma_-

em_mom=compute_empirical_moments_univariate(samples, 2d)

obj_f=dot(em_mom,-qc-sigpc)       
    
sos1=1+q+sigp
model,info1 = add_psatz!(model, sos1, [x], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr1")

sos2=1-q+sigm
model,info2 = add_psatz!(model, sos2, [x], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr2")

exponential_moments=exponential_moments_univariate_up_to(2d, mu)

if trace_penalization
    sos3 = vareps*(1+sum(mu^(2k) for k in 1:d))+sum((qc[i]-sigmc[i])*exponential_moments[i] for i=1:length(qc))   
else
    sos3= sum((qc[i]-sigmc[i])*exponential_moments[i] for i=1:length(qc)) 
end   
model,info3 = add_psatz!(model, sos3, [mu], S, [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr3")

sos4=sigp
model,info4 = add_psatz!(model, sos4, [x], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr4")

sos5=sigm
model,info5 = add_psatz!(model, sos5, [x], [], [], d, QUIET=true, CS=false,TS=false, Groebnerbasis=false, constrs="constr5")


@objective(model, Max, obj_f)
optimize!(model)
objv = objective_value(model)
@show objv

# retrieve moment matrices
infos=[info1,info2,info3,info4,info5]

#moment_list = Vector{Vector{Float64}}(undef, 5)
#MomMat_list = Vector{Matrix{Float64}}(undef, 5)
    
#for k in 1:5
    #info = getfield(Main, Symbol("info$(k)"))  # Only if info1, ..., info5 are global
    #info=infos[k]
    #moment_list[k] = [-dual(constraint_by_name(model, "constr$(k)[$i]")) for i in 1:size(info.tsupp, 2) ]
    #MomMat_list[k] = get_moment_matrix(moment_list[k], info.tsupp, info.cql, info.basis)[1]
#end

    # retrieve moment matrices
moment1 = [-dual(constraint_by_name(model, "constr1[$i]")) for i=1:size(info1.tsupp, 2)]
MomMat1 = get_moment_matrix(moment1, info1.tsupp, info1.cql, info1.basis)
    
moment2 = [-dual(constraint_by_name(model, "constr2[$i]")) for i=1:size(info2.tsupp, 2)]
MomMat2 = get_moment_matrix(moment2, info2.tsupp, info2.cql, info2.basis)
    
moment3 = [-dual(constraint_by_name(model, "constr3[$i]")) for i=1:size(info3.tsupp, 2)]
MomMat3 = get_moment_matrix(moment3, info3.tsupp, info3.cql, info3.basis)
    
moment4 = [-dual(constraint_by_name(model, "constr4[$i]")) for i=1:size(info4.tsupp, 2)]
MomMat4 = get_moment_matrix(moment4, info4.tsupp, info4.cql, info4.basis)
    
moment5 = [-dual(constraint_by_name(model, "constr5[$i]")) for i=1:size(info5.tsupp, 2)]
MomMat5 = get_moment_matrix(moment5, info5.tsupp, info5.cql, info5.basis)

#return objv , MomMat_list, MomMat_list[3],  model, [info1,info3] #GramMat, GramMat2
return objv , [MomMat1,MomMat2,MomMat3,MomMat4,MomMat5], MomMat3,  model, [info1,info3] 
    
end





























########################################################################################################################################
################################################# EXTRACTION
########################################################################################################################################

"""
analyse_relaxations(reslist, d; mdim=2)

reslist[d] must be `[objv, MomMat1, MomMat2, model, [info1, info2]]`.
We compare the parameter-side moment matrix at orders d and d-1.
`mdim` = number of parameter variables (for you: m, sigma → 2).

Returns (mmd, sub_mmd, mmdminus1, basis_KF).
"""
function analyse_relaxations(reslist, d, mdim)
    @assert d ≥ 2 "need at least two orders to compare (d and d-1)"

    # pull parameter-side moment matrices (MomMat2)
    M_d   = reslist[d][3]
    M_dm1 = reslist[d-1][3]
    M_d   = isa(M_d,   AbstractVector)  ? M_d[1]   : M_d
    M_dm1 = isa(M_dm1, AbstractVector)  ? M_dm1[1] : M_dm1

    # pull the corresponding bases from info2
    basis_d   = reslist[d][end][2].basis[1][1]
    basis_dm1 = reslist[d-1][end][2].basis[1][1]

    # your helper to keep the marginal on the first `mdim` variables
    bd,  mmd   = extract_marginal_en_mm(M_d,   basis_d,   mdim)
    bd1, mmd1  = extract_marginal_en_mm(M_dm1, basis_dm1, mdim)

    # align sizes (top-left principal block of the higher-order matrix)
    sub_mmd = mmd[1:size(mmd1,1), 1:size(mmd1,1)]

    #display(mmd1-sub_mmd)

    # eigenvalues (use Symmetric to be safe)
    ev_mmd     = eigvals(Symmetric(mmd))
    ev_sub_mmd = eigvals(Symmetric(sub_mmd))

    # show the largest few for a quick flatness check
    println("Eigenvalues at order $d:")
    println(sort(ev_mmd, rev=true)[1:min(10, length(ev_mmd))])
    println("Eigenvalues of principal block matching order $(d-1):")
    println(sort(ev_sub_mmd, rev=true)[1:min(10, length(ev_sub_mmd))])

    # basis columns kept (first mdim rows = exponents for the kept variables)
    basis_KF = basis_d[1:mdim, bd]

    return mmd, sub_mmd, mmd1, basis_KF
end


function analyse_relaxations_sigma(res, d, mdim)
    # pull parameter-side moment matrices (MomMat2)
    
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
















function i_cols(A, r)
  QR = qr(A, Val(true))
    return QR.p[1:r]
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




function analyse_relaxations_poisson(reslist, d, mdim)
    @assert d ≥ 2 "need at least two orders to compare (d and d-1)"

    # pull parameter-side moment matrices (MomMat2)
    M_d   = reslist[d][3]
    M_dm1 = reslist[d-1][3]
    M_d   = isa(M_d,   AbstractVector)  ? M_d[1]   : M_d
    M_dm1 = isa(M_dm1, AbstractVector)  ? M_dm1[1] : M_dm1

    # pull the corresponding bases from info2
    basis_d   = reslist[d][end][2].basis[1][1]
    basis_dm1 = reslist[d-1][end][2].basis[1][1]

    # your helper to keep the marginal on the first `mdim` variables
    #bd,  mmd   = extract_marginal_en_mm(M_d,   basis_d,   mdim)
    #bd1, mmd1  = extract_marginal_en_mm(M_dm1, basis_dm1, mdim)

    # align sizes (top-left principal block of the higher-order matrix)
    sub_mmd = M_d[1:d, 1:d]

    #display(mmd1-sub_mmd)

    # eigenvalues (use Symmetric to be safe)
    ev_mmd     = eigvals(Symmetric(M_d))
    ev_sub_mmd = eigvals(Symmetric(sub_mmd))

    # show the largest few for a quick flatness check
    println("Eigenvalues at order $d:")
    println(sort(ev_mmd, rev=true)[1:min(10, length(ev_mmd))])
    println("Eigenvalues of principal block matching order $(d-1):")
    println(sort(ev_sub_mmd, rev=true)[1:min(10, length(ev_sub_mmd))])

    # basis columns kept (first mdim rows = exponents for the kept variables)
    basis_KF = basis_d

    return M_d, sub_mmd, M_dm1, basis_KF
end










#########################







