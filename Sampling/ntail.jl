function ntail(l,u)
    # samples a column vector of length=length(l)=length(u)
    # from the standard multivariate normal distribution
    # truncated over the region [l,u], where l>0 &
    # l and u are column vectors
    # uses acceptance-rejection from Rayleigh distr.
    # similar to Marsaglia [1964]
    c = l.^2/2
    n = length(l)
    f = exp.(c - u.^2/2) .- 1
    x = c - log.(1 .+ rand(n).*f) # sample using Rayleigh
    # keep list of rejected
    I = findall(rand(n).^2 .*x .> c)
    d = length(I)
    while d > 0 # while there are rejections
        cy = c[I] # find the thresholds of rejected
        y = cy - log.(1 .+ rand(d).*f[I])
        idx = rand(d).^2 .*y .< cy # accepted
        x[I[idx]] = y[idx] # store the accepted
        I = I[.~idx] # remove accepted from list
        d = length(I) # number of rejected
    end
    x = sqrt.(2*x) # this Rayleigh transform can be delayed till the end
    return x
end
