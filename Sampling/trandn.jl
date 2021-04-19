function trandn(l,u)
    ## truncated normal generator
    # * efficient generator of a vector of length(l)=length(u)
    # from the standard multivariate normal distribution
    # truncated over the region [l,u]
    # infinite values for "u' & 'l" are accepted
    # * Remark:
    # If you wish to simulate a random variable
    # 'Z' from the non-standard Gaussian N[m,s^2]
    # conditional on l<Z<u; then first simulate
    # X=trandn((l-m)/s,(u-m)/s) & set Z=m+s*X
    #
    # See also: norminvp
    #
    # For more help; see <a href="matlab:
    # doc">Truncated Multivariate Student & Normal</a> documentation at the bottom.
    # Reference: Z. I. Botev [2017], _The Normal Law Under Linear Restrictions:
    # Simulation & Estimation via Minimax Tilting_; Journal of the Royal
    # Statistical Society; Series B; Volume 79; Part 1; pp. 1-24
    if any(l .> u)
        error("Truncation limits have to be vectors of the same length with l<u")
    end
    sizel = length(l)
    x = fill(NaN, sizel)
    a = .66 # treshold for switching between methods()
    # threshold can be tuned for maximum speed for each Matlab version()
    # three cases to consider:
    # case 1: a<l<u
    I = l .> a
    if any(I)
        tl = l[I]
        tu = u[I]
        x[I] = ntail(tl,tu)
    end
    # case 2: l<u<-a
    J = u .< -a
    if any(J)
        tl = -u[J]
        tu = -l[J]
        x[J] = -ntail(tl,tu)
    end
    # case 3: otherwise use inverse transform | accept-reject
    I = .~(I .| J)
    if  any(I)
        tl = l[I]
        tu = u[I]
        x[I] = tn(tl,tu)
    end
    return x
end
