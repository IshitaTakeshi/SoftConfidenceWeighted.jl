module SoftConfidenceWeighted

import Distributions: Normal, cdf

export init, fit!, predict, SCW1, SCW2


typealias AA AbstractArray

@enum SCWType SCW1 SCW2


type CDF
    ϕ
    ψ
    ζ

    function CDF(ETA)
        ϕ = cdf(Normal(0, 1), ETA)  # CDF of the standard normal distribution
        ψ = 1 + ϕ^2 / 2
        ζ = 1 + ϕ^2
        new(ϕ, ψ, ζ)
    end
end


#calc cdf in a constructor
type SCW
    C::Float64
    cdf::CDF
    ndim::Int64
    w::Array{Float64, 1}  # weights
    Σ::Array{Float64, 2}  # covariance
    has_fitted::Bool

    function SCW(C, ETA)
        new(C, CDF(ETA), -1, zeros(0), zeros(0, 0), false)
    end
end


function set_dimension!(scw::SCW, ndim::Int)
    assert(!scw.has_fitted)
    scw.ndim = ndim
    scw.w = zeros(ndim)
    scw.Σ = eye(ndim)
    scw.has_fitted = true
    return scw
end


calc_margin{T<:AA,R<:Real}(scw::SCW, x::T, y::R) = y * dot(scw.w, x)


calc_confidence{T<:AA}(scw::SCW, x::T) = (x' * scw.Σ * x)[1]


function calc_α1{T<:AA,R<:Real}(scw::SCW, x::T, y::R)
    v = calc_confidence(scw, x)
    m = calc_margin(scw, x, y)
    cdf = scw.cdf

    j = m^2 * cdf.ϕ^4 / 4
    k = v * cdf.ζ * cdf.ϕ^2
    t = (-m*cdf.ψ + √(j+k)) / (v*cdf.ζ)
    return min(scw.C, max(0, t))
end


function calc_α2{T<:AA,R<:Real}(scw::SCW, x::T, y::R)
    v = calc_confidence(scw, x)
    m = calc_margin(scw, x, y)
    cdf = scw.cdf

    n = v + 1 / 2scw.C
    a = (cdf.ϕ*m*v)^2
    b = 4*n*v * (n + v * cdf.ϕ^2)
    γ = cdf.ϕ * √(a+b)

    c = -(2 * m * n + m * v * cdf.ϕ^2)
    d = n^2 + n * v * cdf.ϕ^2
    t = (c+γ) / 2d
    return max(0, t)
end


function init(;C = 1, ETA = 1, type_ = SCW1::SCWType)
    global calc_α
    if type_ == SCW1
        calc_α = calc_α1
    elseif type_ == SCW2
        calc_α = calc_α2
    else
        assert(true)
    end

    return SCW(C, ETA)
end


function loss{T<:AA,R<:Real}(scw::SCW, x::T, y::R)
    t = y * dot(scw.w, x)
    if t >= 1
        return 0
    end
    return 1-t
end


function calc_β{T<:AA,R<:Real}(scw::SCW, x::T, y::R)
    α = calc_α(scw, x, y)
    v = calc_confidence(scw, x)
    m = calc_margin(scw, x, y)
    cdf = scw.cdf

    j = -α * v * cdf.ϕ
    k = √((α*v*cdf.ϕ)^2 + 4v)
    u = (j+k)^2 / 4
    return (α * cdf.ϕ) / (√u + v*α*cdf.ϕ)
end


function update_Σ!{T<:AA,R<:Real}(scw::SCW, x::T, y::R)
    β = calc_β(scw, x, y)

    # same as
    # scw.Σ -= β * Σ * x * x' * Σ
    BLAS.axpy!(-β, scw.Σ * x * x' * scw.Σ, scw.Σ)
    return scw
end


function update_w!{T<:AA,R<:Real}(scw::SCW, x::T, y::R)
    α = calc_α(scw, x, y)

    # same as
    # scw.w += α * y * (scw.Σ .* x)
    BLAS.axpy!(α * y, scw.Σ * x, scw.w)
    return scw
end


function update!{T<:AA,R<:Real}(scw::SCW, x::T, y::R)
    x = vec(full(x))
    if loss(scw, x, y) > 0
        update_w!(scw, x, y)
        update_Σ!(scw, x, y)
    end
    return scw
end


function train!{T<:AA,R<:AA}(scw::SCW, X::T, ys::R)
    # preallocate for performance optimization
    for i in 1:size(X, 2)
        y = ys[i]

        if y != 1 && y != -1
            throw(ArgumentError("Label must be 1 or -1"))
        end

        update!(scw, view(X, :, i), y)
    end
    return scw
end


function fit!{T, R}(scw::SCW, X::AA{T, 2}, ys::AA{R, 1})
    if size(X, 2) != length(ys)
        message = "X and y have incompatible shapes. " *
                  "size(X) = $(size(X)) size(y) = $(size(y))"
        throw(ArgumentError(message))
    end

    if !scw.has_fitted
        ndim = size(X, 1)
        set_dimension!(scw, ndim)
    end

    train!(scw, X, ys)
    return scw
end


function fit!{T}(scw::SCW, x::AA{T, 1}, y::Real)
    ndim = size(x, 1)
    if !scw.has_fitted
        set_dimension!(scw, ndim)
    end

    update!(scw, x, y)
end


@deprecate fit fit!


function compute{T<:AA}(scw::SCW, x::T)
    x = vec(full(x))
    if dot(x, scw.w) > 0
        return 1
    else
        return -1
    end
end


function predict{T<:AA}(scw::SCW, X::T)
    N = size(X, 2)
    ys = Array(Int, N)
    for i in 1:size(X, 2)
        ys[i] = compute(scw, view(X, :, i))
    end
    return ys
end

end # module
