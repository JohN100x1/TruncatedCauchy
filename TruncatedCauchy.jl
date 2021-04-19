using LinearAlgebra
using SpecialFunctions
using Distributions
using PDMats
using Cuba
#using HCubature
using Random

import Base: length
import Distributions: sqmahal, rand, pdf

include("./Sampling/mvrandn.jl")

abstract type AbstractTruncatedCauchy <: ContinuousMultivariateDistribution end

struct GenericTruncatedCauchy{L<:Real, Cov<:AbstractPDMat, Mean<:AbstractVector} <: AbstractTruncatedCauchy
    limit::L
    dim::Int
    μ::Mean
    Σ::Cov

    function GenericTruncatedCauchy{L,Cov,Mean}(limit::L, dim::Int, μ::Mean, Σ::AbstractPDMat) where {L,Cov,Mean}
        new{L,Cov,Mean}(limit, dim, μ, Σ)
    end
end

function GenericTruncatedCauchy(limit::L, μ::Mean, Σ::Cov) where {Cov<:AbstractPDMat, Mean<:AbstractVector, L<:Real}
    d = length(μ)
    R = Base.promote_eltype(L, μ, Σ)
    S = convert(AbstractArray{R}, Σ)
    m = convert(AbstractArray{R}, μ)
    GenericTruncatedCauchy{R, typeof(S), typeof(m)}(R(limit), d, m, S)
end

## Construction
TruncatedCauchy(L::Real, μ::Vector{<:Real}, Σ::Matrix{<:Real}) = GenericTruncatedCauchy(L, μ, PDMat(Σ))

length(d::GenericTruncatedCauchy) = d.dim
sqmahal(d::GenericTruncatedCauchy, x::AbstractVector{<:Real}) = invquad(d.Σ, x - d.μ)
inlimits(d::GenericTruncatedCauchy, x::AbstractVector{<:Real}) = all(x .> d.μ .- d.limit) && all(x .< d.μ .+ d.limit) ? true : false

function pdf(d::AbstractTruncatedCauchy, X::AbstractVector)
    length(X) == length(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _pdf(d, X)
end

function _pdf(d::AbstractTruncatedCauchy, x::AbstractVector{T}) where T<:Real
    # Check if x lies in the truncated region
    if inlimits(d, x)
        a = d.μ .- d.limit
        b = d.μ .+ d.limit
        function integrand(y, f)
            f[1] = prod(b-a) * _updf(d, a .+ (b-a).*y)
        end
        c = gamma((1+d.dim)/2)/(gamma(0.5) * pi^(0.5*d.dim) * det(d.Σ)^(0.5))
        f = c * _updf(d, x) ./ cuhre(integrand, d.dim)[1][1]
    else
        f = 0
    end
    return f
end


function _updf(d::AbstractTruncatedCauchy, x::AbstractVector{T}) where T<:Real
    return (1 ./ (1 .+ sqmahal(d, x)).^((1+d.dim)/2))
end

# Sampling (for TruncatedCauchy)
function _rand!(rng::AbstractRNG, d::GenericTruncatedCauchy, x::AbstractVector{<:Real})
    println(d.μ .- d.limit)
    println(d.Σ)
    return mvrandn(d.μ .- d.limit, d.μ .+ d.limit, d.Σ, 1, 1)
end

function _rand!(rng::AbstractRNG, d::GenericTruncatedCauchy, x::AbstractMatrix{T}) where T<:Real
    cols = size(x,2)
    return mvrandn(d.μ .- d.limit, d.μ .+ d.limit, d.Σ[:,:], 1, cols)
end

# multiple multivariate, must allocate matrix or array of vectors
rand(d::GenericTruncatedCauchy, n::Int) = rand(Random.GLOBAL_RNG, d, n)
rand(rng::AbstractRNG, d::GenericTruncatedCauchy, n::Int) = _rand!(rng, d, Matrix{eltype(d)}(undef, length(d), n))
rand(rng::AbstractRNG, d::GenericTruncatedCauchy, dims::Dims) = rand(rng, d, Array{Vector{eltype(d)}}(undef, dims), true)

# single multivariate, must allocate vector
rand(rng::AbstractRNG, d::GenericTruncatedCauchy) = _rand!(rng, d, Vector{eltype(d)}(undef, length(d)))
