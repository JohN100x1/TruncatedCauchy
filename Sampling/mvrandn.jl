using LinearAlgebra
using SpecialFunctions
using NLsolve
include("lnNpr.jl")
include("lnPhi.jl")
include("cholperm.jl")
include("gradpsiT.jl")
include("psyT.jl")
include("tn.jl")
include("trnd.jl")
include("trandn.jl")
include("mvtrnd.jl")
include("ntail.jl")

function mvrandn(l,u,Sig,df,n)
    ## truncated multivariate student generator
    # simulates 'n' random vectors exactly/perfectly distributed
    # from the d-dimensional t_nu[0,Sig] distribution (zero-mean student
    # with scale matrix "Sig" & deg. of freedom df); conditional on l<X<u
    # infinite values for "l' & 'u" are accepted
    # output:   "d' times 'n' array 'rv" storing random vectors
    #
    # * Example:
    #  d=60;n=10^3;Sig=0.9*ones(d,d)+0.1*I;l=(1:d)/d*4;u=l+2; df=10
    #  X=mvrandt[l,u,Sig,df,n];boxplot[X",'plotstyle','compact"] # plot marginals
    #
    # * Notes: Algorithm may not work if "Sig" is close to being rank deficient.
    #
    # See also: mvNcdf; mvNqmc; mvTcdf; mvTqmc; mvrorth
    #
    # For more help; see <a href="matlab:
    # doc">Truncated Multivariate Student & Normal</a> documentation at the bottom.
    # References:
    # [1] Z. I. Botev [2017], _The Normal Law Under Linear Restrictions:
    # Simulation & Estimation via Minimax Tilting_; Journal of the Royal
    # Statistical Society; Series B; Volume 79; Part 1; pp. 1-24
    #
    # [2] Z. I. Botev & P. L'Ecuyer [2015], _EFFICIENT PROBABILITY ESTIMATION
    # AND SIMULATION OF THE TRUNCATED MULTIVARIATE STUDENT-t DISTRIBUTION_;
    # Proceedings of the 2015 Winter Simulation Conference; pages 380-391;
    # (L. Yilmaz, W. Chan, I. Moon, T. Roeder, C. Macal, & M. Rossetti, eds.)
    d = length(l) # basic input check
    if  size(Sig)[1] != size(Sig)[2] | (d != size(Sig)[1] | any(l .> u))
        error("l, u, & Sig have to match in dimension with u>l")
    end
    # Cholesky decomposition of matrix with permuation
    Lfull, l, u, perm = cholperm(Sig,l,u) # outputs the permutation
    D = diag(Lfull)
    if any(D .< eps(1.0))
        @warn("Method may fail as covariance matrix is singular!")
    end
    # rescale
    L = Lfull./repeat(D,1,d)
    u = u./D
    l = l./D
    L = L - I # remove diagonal
    # find optimal tilting parameter non-linear equation solver
    x0 = zeros(2*d,1)
    x0[2*d] = sqrt(df)
    x0[d] = log(sqrt(df))
    f(x) = gradpsiT(x,L,l,u,df)
    soln = nlsolve(f,x0).zero
    # assign saddlepoint x* & mu*
    soln[d] = exp(soln[d])
    x = soln[1:d]
    mu = soln[(d+1):end]
    # compute psi star
    psistar = psyT(x,L,l,u,df,mu)
    # start acceptance rejection sampling
    rv = Array{Float64}(undef,2,0)
    accept = 0
    iter = 0
    while accept < n # while # of accepted is less than n
        logpr, Z, R = mvtrnd(n,L,l,u,df,mu) # simulate n proposals
        Z = sqrt(df)*Z./repeat(R,d,1) # deliver a student transformation
        idx = -log.(rand(n)) .> (psistar .- logpr) # acceptance tests
        if any(idx)
            rv = hcat(rv,Z[:,idx])  # accumulate accepted
        end
        accept = size(rv,2) # keep track of # of accepted
        iter = iter+1  # keep track of while loop iterations
        if iter == 10^3 # if iterations are getting large; give warning()
            @warn("Acceptance prob. smaller than 0.001")
        elseif iter > 10^4 # if iterations too large; seek approximation only
            accept = n
            rv = hcat(rv,Z) # add the approximate samples
            @warn("Sample is only approximately distributed.")
        end
    end
    # finish sampling; postprocessing
    order = sortperm(perm)
    rv = rv[:,1:n] # cut-down the array to desired n samples
    rv = Lfull*rv # reverse scaling of L
    rv = rv[order,:] # reverse the Cholesky permutation
    return rv
end
