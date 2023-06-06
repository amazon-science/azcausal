

library(synthdid)




sum_normalize = function(x) {
    if(sum(x) != 0) { x / sum(x) }
    else { rep(1/length(x), length(x)) }
    # if given a vector of zeros, return uniform weights
    # this fine when used in bootstrap and placebo standard errors, where it is used only for initialization
    # for jackknife standard errors, where it isn't, we handle the case of a vector of zeros without calling this function.
}

placebo_sample = function(estimate, replications) {
    setup = attr(estimate, 'setup')
    opts = attr(estimate, 'opts')
    weights = attr(estimate, 'weights')
    N1 = nrow(setup$Y) - setup$N0
    if (setup$N0 <= N1) { stop('must have more controls than treated units to use the placebo se') }
    theta = function(ind) {
	N0 = length(ind)-N1
	weights.boot = weights
	weights.boot$omega = sum_normalize(weights$omega[ind[1:N0]])
        do.call(synthdid_estimate, c(list(Y=setup$Y[ind,], N0=N0,  T0=setup$T0,  X=setup$X[ind, ,], weights=weights.boot), opts))
    }
    replicate(replications, theta(sample(1:setup$N0)))
}

