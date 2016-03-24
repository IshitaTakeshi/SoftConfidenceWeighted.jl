module SoftConfidenceWeighted

using Devectorize

import Distributions: Normal, cdf
import SVMLightLoader: SVMLightFile

export init, fit!, predict, SCW1, SCW2


typealias AA AbstractArray

@enum SCWType SCW1 SCW2


type CDF
    phi
    psi
    zeta

    function CDF(ETA)
        phi = cdf(normal_distribution, ETA)
        psi = 1 + phi^2 / 2
        zeta = 1 + phi^2
        new(phi, psi, zeta)
    end
end


#calc cdf in a constructor
type SCW
    C::Float64
    cdf::CDF
    ndim::Int64
    weights::Array{Float64, 1}
    covariance::Array{Float64, 1}
    has_fitted::Bool

    function SCW(C, ETA)
        new(C, CDF(ETA), -1, [], [], false)
    end
end


function set_dimension!(scw::SCW, ndim::Int)
    assert(!scw.has_fitted)
    scw.ndim = ndim
    scw.weights = zeros(ndim)
    scw.covariance = ones(ndim)
    scw.has_fitted = true
    return scw
end


normal_distribution = Normal(0, 1)


function calc_margin{T<:AA,R<:Real}(scw::SCW, x::T, label::R)
    #Devectorize.jl requires assignment
    @devec t = label .* dot(scw.weights, x)
end


function calc_confidence{T<:AA}(scw::SCW, x::T)
    @devec t = dot(x, scw.covariance .* x)
end


function calc_alpha1{T<:AA,R<:Real}(scw::SCW, x::T, label::R)
    v = calc_confidence(scw, x)
    m = calc_margin(scw, x, label)
    cdf = scw.cdf
    (phi, psi, zeta) = (cdf.phi, cdf.psi, cdf.zeta)

    j = m^2 * phi^4 / 4
    k = v * zeta * phi^2
    t = (-m*psi + sqrt(j+k)) / (v*zeta)
    return min(scw.C, max(0, t))
end


function calc_alpha2{T<:AA,R<:Real}(scw::SCW, x::T, label::R)
    v = calc_confidence(scw, x)
    m = calc_margin(scw, x, label)
    cdf = scw.cdf
    (phi, psi, zeta) = (cdf.phi, cdf.psi, cdf.zeta)

    n = v+1 / scw.C
    a = (phi*m*v)^2
    b = 4*n*v * (n + v * phi^2)
    gamma = phi * sqrt(a+b)

    c = -(2 * m * n + m * v * phi^2)
    d = n^2 + n * v * phi^2
    t = (c+gamma) / 2d
    return max(0, t)
end


function init(;C = 1, ETA = 1, type_ = SCW1::SCWType)
    global calc_alpha
    if type_ == SCW1
        calc_alpha = calc_alpha1
    elseif type_ == SCW2
        calc_alpha = calc_alpha2
    else
        assert(true)
    end

    return SCW(C, ETA)
end


function loss{T<:AA,R<:Real}(scw::SCW, x::T, label::R)
    @devec t = label .* dot(scw.weights, x)
    if t >= 1
        return 0
    end
    return 1-t
end


function calc_beta{T<:AA,R<:Real}(scw::SCW, x::T, label::R)
    alpha = calc_alpha(scw, x, label)
    v = calc_confidence(scw, x)
    m = calc_margin(scw, x, label)
    cdf = scw.cdf
    (phi, psi, zeta) = (cdf.phi, cdf.psi, cdf.zeta)

    j = -alpha * v * phi
    k = sqrt((alpha*v*phi)^2 + 4v)
    u = (j+k)^2 / 4
    return (alpha * phi) / (sqrt(u) + v*alpha*phi)
end


function update_covariance!{S<:AA,T<:AA,R<:Real}(t::S, scw::SCW, x::T, label::R)
    beta = calc_beta(scw, x, label)
    c = scw.covariance

    # same as
    # scw.covariance -= beta * (c .* x) .* (c .* x)
    @devec t[:] = (c .* x) .* (c .* x)
    BLAS.axpy!(-beta, t, scw.covariance)
    return scw
end


function update_weights!{S<:AA,T<:AA,R<:Real}(t::S, scw::SCW, x::T, label::R)
    alpha = calc_alpha(scw, x, label)

    # same as
    # scw.weights += alpha * label * (scw.covariance .* x)
    @devec t[:] = scw.covariance .* x
    BLAS.axpy!(alpha * label, t, scw.weights)
    return scw
end


function update!{S<:AA,T<:AA,R<:Real}(t::S, scw::SCW, x::T, label::R)
    x = vec(full(x))
    if loss(scw, x, label) > 0
        update_weights!(t, scw, x, label)
        update_covariance!(t, scw, x, label)
    end
    return scw
end


function train!{T<:AA,R<:AA}(scw::SCW, X::T, labels::R)
    # preallocate for performance optimization
    t = Array(Float64, size(X, 1))
    for i in 1:size(X, 2)
        label = labels[i]

        if label != 1 && label != -1
            throw(ArgumentError("Label must be 1 or -1"))
        end

        update!(t, scw, slice(X, :, i), label)
    end
    return scw
end


function fit!{T, R}(scw::SCW, X::AA{T, 2}, labels::AA{R, 1})
    if size(X, 2) != length(labels)
        message = "X and y have incompatible shapes. " *
                  "size(X) = $(size(X)) size(y) = $(size(y))"
        throw(ArgumentError(message))
    end

    if !scw.has_fitted
        ndim = size(X, 1)
        set_dimension!(scw, ndim)
    end

    train!(scw, X, labels)
    return scw
end


function fit!{T}(scw::SCW, x::AA{T, 1}, label::Real)
    ndim = size(x, 1)
    if !scw.has_fitted
        set_dimension!(scw, ndim)
    end

    t = Array(Float64, ndim)
    update!(t, scw, x, label)
end


function fit!(scw::SCW, filename::AbstractString, ndim::Int64)
    if !scw.has_fitted
        set_dimension!(scw, ndim)
    end

    t = Array(Float64, ndim)
    for (vector, label) in SVMLightFile(filename, ndim)
        update!(t, scw, vector, label)
    end
    return scw
end


@deprecate fit fit!


function compute{T<:AA}(scw::SCW, x::T)
    x = vec(full(x))
    if dot(x, scw.weights) > 0
        return 1
    else
        return -1
    end
end


function predict{T<:AA}(scw::SCW, X::T)
    N = size(X, 2)
    labels = Array(Int, N)
    for i in 1:size(X, 2)
        labels[i] = compute(scw, slice(X, :, i))
    end
    return labels
end


function predict(scw::SCW, filename::AbstractString)
    labels = Int64[]
    for (x, _) in SVMLightFile(filename, scw.ndim)
        label = compute(scw, x)
        push!(labels, label)
    end
    return labels
end

end # module
