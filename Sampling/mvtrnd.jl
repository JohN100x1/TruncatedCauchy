function mvtrnd(n,L,l,u,nu,mu)
    # generates the proposals from the exponentially tilted
    # sequential importance sampling pdf
    # output:    'p'; log-likelihood of sample
    #             Z; Gaussian sample
    #             R, random scale parameter so that sqrt(nu)*Z/R is student
    d = length(l) # Initialization
    eta = mu[d]
    mu[d] = 0
    Z = zeros(d,n) # create array for variables
    # precompute constants
    c = log(2*pi)/2-log(gamma(nu/2))-(nu/2-1)*log(2)+lnNpr([-eta],[Inf])[1]+.5*eta^2
    R = eta .+ trandn(-eta*ones(n),fill(Inf,n))' # simulate R~N[eta,1] with R>0
    p = ((nu-1)*log.(R) - eta*R .+ c)[1,:] # compute Likelihood Ratio for R
    #R=R/sqrt(nu); # scale parameter divided by nu
    for k = 1:d
        # compute matrix multiplication L*Z
        col = L[k,1:k]'*Z[1:k,:]
        # compute limits of truncation
        tl = (R*l[k]/sqrt(nu) .- mu[k] .- col)[1,:]
        tu = (R*u[k]/sqrt(nu) .- mu[k] .- col)[1,:]
        #simulate N[mu,1] conditional on [tl,tu]
        Z[k,:] = mu[k] .+ trandn(tl,tu)
        # update likelihood ratio
        p = p .+ lnNpr(tl,tu) .+ .5*mu[k]^2 .- mu[k]*Z[k,:]
    end
    return p, Z, R
end
