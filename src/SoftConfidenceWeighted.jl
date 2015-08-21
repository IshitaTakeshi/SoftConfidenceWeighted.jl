module SoftConfidenceWeighted

import Distributions: Normal, cdf

export init, fit, predict, SCW1, SCW2


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
    weights::Array{Float64, 1}
    covariance::Array{Float64, 1}
    has_fitted::Bool

    function SCW(C, ETA)
        new(C, CDF(ETA), [], [], false)
    end
end


normal_distribution = Normal(0, 1)


function calc_margin(scw, x, label)
    return label * dot(scw.weights, x)
end


function calc_confidence(scw, x)
    return dot(x, (scw.covariance .* x))
end


function calc_alpha1(scw, x, label)
    v = calc_confidence(scw, x)
    m = calc_margin(scw, x, label)
    cdf = scw.cdf
    (phi, psi, zeta) = (cdf.phi, cdf.psi, cdf.zeta)

    j = m^2 * phi^4 / 4
    k = v * zeta * phi^2
    t = (-m*psi + sqrt(j+k)) / (v*zeta)
    return min(scw.C, max(0, t))
end


function calc_alpha2(scw, x, label)
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


function init(C, ETA, type_=SCW1::SCWType)
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


function loss(scw, x, label)
    t = label * dot(scw.weights, x)
    if t >= 1
        return 0
    end
    return 1-t
end


function calc_beta(scw, x, label)
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


function update_covariance(scw, x, label)
    beta = calc_beta(scw, x, label)
    c = scw.covariance
    scw.covariance -= beta * (c .* x) .* (c .* x)
    return scw
end


function update_weights(scw, x, label)
    alpha = calc_alpha(scw, x, label)
    scw.weights += alpha * label * (scw.covariance .* x)
    return scw
end


function update(scw::SCW, x, label)
    if loss(scw, x, label) > 0
        scw = update_weights(scw, x, label)
        scw = update_covariance(scw, x, label)
    end
    return scw
end


function train(scw, X, labels)
    for i in 1:size(X, 1)
        x = vec(full(X[i, :]))
        scw = update(scw, x, labels[i])
    end
    return scw
end


function fit(scw::SCW, X, labels)
    assert(ndims(X) == 2)
    assert(ndims(labels) == 1)
    ndim = size(X)[2]
    if !scw.has_fitted
        scw.weights = zeros(ndim)
        scw.covariance = ones(ndim)
        scw.has_fitted = true
    end
    scw = train(scw, X, labels)
    return scw
end


function compute(scw, x)
    if dot(x, scw.weights) > 0
        return 1
    else
        return -1
    end
end


function predict(scw, X)
    return [compute(scw, vec(full(X[i, :]))) for i in 1:size(X, 1)]
end

end # module
