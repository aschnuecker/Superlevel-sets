module SLS

using Reexport
using Plots
using Distributions
using LinearAlgebra
using Plots
using Optim, ImplicitEquations
using JuMP, Gurobi

@reexport using DistributionTools
export plotSLS, plotSLS!, ellipsoid_quantile, CenterOutward!, DirectionalKB

# Quantile regression
"""
    quantilereg(y, X, τ)

Returns coefficients of the quantile regression of `y` on `X` at quantile `τ`
"""
function quantilereg(y::AbstractVector, X::AbstractMatrix, τ::AbstractFloat)
    f(β) = mean(τ*max.(y-X*β, zeros(size(y))) + (1-τ)*max.(X*β-y, zeros(size(y))))
    beta = zeros(size(X,2))
    r = optimize(f, beta, BFGS(); autodiff=:forward)
    sol = Optim.minimizer(r)
    
    return sol
end


# Ellipsoid method
""" 
    ellipsoid_quantile(data, tau)

Returns the implicit equation of the smallest ellipsoid that contains fraction `tau` of the points in `data` 
by directly minimizing the corresponding check function
"""
function ellipsoid_quantile(data, tau)

    # Transform p11, p12 (elements of choleski) to matrix A
    function getA(p11,p12) 
        P =  [p11 p12; 0 1/p11]
        return P'P
    end

    # Objective function to mimimize
    function min_func(par) 
        A     = getA(par[1], par[2])
        s     = par[3:4]
        kappa = par[5]
        
        g(v::AbstractVector{Float64}) = dot(v-s,A,v-s) - kappa           
        
        r = mean( tau*gd - (gd .< 0).*gd for gd in g.(eachcol(data)))                

        return r
    end

    mu = mean(data, dims=2)
    
    par = [1.0; 0; mu[1]; mu[2]; 1]     # some sensible starting value
    
    # Minimize objective 
    r = optimize(min_func, par, BFGS(),Optim.Options(show_trace = true); autodiff = :forward)
    par = Optim.minimizer(r)

    # Form the solution
    A = getA(par[1], par[2])
    s = par[3:4]
    kappa = par[5]
   
    return ( (x,y) -> ([x; y]-s)'A*([x; y]-s), kappa, s)
end

# Directional Koenker Bassett method
"""
    OrthBasis(u)

Returns orthonormal basis vectors of the space orthogonal to vector `u`
"""
function OrthBasis(u::AbstractVector)
    @assert u'u ≈ 1 "`u` needs to be length 1"
           
    basis = I - u*u'
    
    # Drop one of the vectors
    sel = [basis[:,k] == zeros(length(u)) for k in 1:length(u)]
    if sum(sel)==0        
        @assert isapprox(det(basis), 0.0, atol=1e-10) "Should not happen"        
        basis = basis[:, 2:end]
    else        
        basis = basis[:, sel.==0]
    end
    
    # Normalize
    basis = hcat([b ./ norm(b) for b in eachcol(basis)]...)
    # Orthoganalize    
    for k in 2:size(basis,2)
        for j in 1:(k-1)
            basis[:,k] = basis[:,k] - ((basis[:,j]'basis[:,k]))/(basis[:,j]'basis[:,j]) * basis[:,j]
        end
        basis[:,k] = basis[:,k] ./ sqrt(basis[:,k]'basis[:,k])
    end
    return basis
end

""" 
    Transform(G::AbstractMatrix, d::MvNormal)

Return the distribution of `G`*x, where x~`d` and d is a MvNormal
"""
function Transform(G::AbstractMatrix, d::MvNormal)
    if size(G,1)>1
        return MvNormal(G*d.μ, Symmetric(G*d.Σ*G'))
    else
        return Normal(G*d.μ, sqrt(G*d.Σ*G'))
    end
end

""" 
    Transform(G::AbstractMatrix, d::MixtureModel{MvNormal})

Return the distribution of `G`*x, where x~`d` and d is a Mixture of MvNormals
"""
function Transform(G::AbstractMatrix, d::MixtureModel{Multivariate,Continuous,<:MvNormal,<:DiscreteNonParametric})
    K = length(d.prior.p)

    return MixtureModel([Transform(G, comp) for comp in d.components], d.prior.p)
end

"""
    DirectionalKB(d, data, τ, step=0.01π)

Plot the directional Koenker Bassett bivariate quantile of `d` at level `τ`, that is, plot half spaces such that u'Z < τ (for various choices of u)

Internally quantile regression is used for each direction u. A scatter is added for the first (max) 1_000 draws.
"""
function DirectionalKB(d::Distribution, data, τ, step=0.01π)
    @assert length(d) == 2 "Needs two-dimensional distribution"
    if length(d)>2
        @warn "Using only first two dimensions"
    end    
    
    p1 = scatter(data[1:min(1000,size(data,1)),1], 
                 data[1:min(1000,size(data,1)),2],
                 label="", 
                 markeropacity=0.4, 
                 markersize=3)
        
    for θ in 0:step:2π        
        # Create direction
        u = [cos(θ), sin(θ)]
        u = u ./ sqrt(u'u)
        
        # Form complementing orthonormal basis
        G = OrthBasis(u)
        
        # Trans
        td = Transform([u';G'], d) 

        low  = my_quantile(marginal(td,2),0.001)
        high = my_quantile(marginal(td,2),0.999)
               
        β = quantilereg(data*u, [ones(size(data,1)) data*G], τ)
        alt_point = vcat([[ [1 x]*β x] for x in low:0.1:high]...)*inv([u G])
        
        lab = θ == 0 ? "Directional KB" : ""        # Add label only for first direction
        plot!(p1, alt_point[:,1], alt_point[:,2], label=lab, linewidth=2, linecolor=:grey)      
    end    
    
    return p1
end

"""
    CenterOutward!(pl::Plots.Plot, Y; probs = [0.2, 0.4, 0.8], Nr = 10-1, Ns=50, kwargs...)

Add center outward quantile to an existing plot. 
* `Y` is a 2xN data matrix
* `probs` sets the desired τ levels (defaults to [0.2, 0.4, 0.8])
* `Nr` is number of radii, or circles to consider (defaults to 9=10-1)
* `Ns` is number of points on each circle, or rays from origin (defaults to 50)
* `kwargs` are forwarded to the plot()

Uses Gurobi to solve the underlying Optimal Transport problem (using LP solver)
"""
function CenterOutward!(pl::Plots.Plot, Y::AbstractMatrix{Float64}; probs = [0.2, 0.4, 0.8], Nr = 10-1, Ns=50, kwargs...)   
    Ndata = size(Y,2)
    
    # Construct the full grid of points
    rays   = range(0, 2π-(2π/Ns), Ns)
    points = [sin.(rays) cos.(rays)]
    radii  = (1:Nr)/(Nr+1)         
    grid = reduce(vcat, [r*points for r in radii] )'    
    grid = [[0;0] grid]
    
    # Optional weight vector
    weight = Vector{Float64}(undef,Ndata)    
    weight .= 1/Ndata           # can adjust to make things local to a certain data point
    
    N = Nr*Ns + 1
    @assert N == size(grid, 2)
    
    # Obtain all squared distances between points in Y and points in the grid
    sqdist = [ sum((y .- g).^2) for y in eachcol(Y), g in eachcol(grid) ]
    
    # Setup Optimal Transportation LP model
    @info "Creating model"
    m = Model(Gurobi.Optimizer) 
    set_attribute(m, "Method", 0)       # select method (https://www.gurobi.com/documentation/current/refman/method.html#parameter:Method)
    set_attribute(m, "Presolve", 0)     # disable presolve as it does not help (https://www.gurobi.com/documentation/current/refman/presolve.html)
    @variable(m, pi[1:Ndata, 1:N], lower_bound=0.0)    

    @objective(m, Min, sum( sqdist[i,j]*pi[i, j] for i in 1:Ndata, j in 1:N) )
    
    @constraint(m, total_weight_gridpoint[j in 1:N], 
                   sum( pi[i,j] for i in 1:Ndata ) == 1.0 )
    @constraint(m, total_weight_observation[i in 1:Ndata], 
                   sum( pi[i,j] for j in 1:N ) == N*weight[i] )

    @info "Solving optimal transport LP model (be patient)"
    optimize!(m)
    @info "Done" 
    optAlloc = value.(pi) 
    
    # Find "best" observation for each grid point
    Tstar = Matrix{Float64}(undef,2,N)

    for (tstar,c) in zip(eachcol(Tstar), eachcol(optAlloc))
        sel = Y[:, c .== maximum(c)] 
        if size(sel,2)>1
        # Multiple points are equally good => find smallest point in convec hull
            @info "Need to use ConvexHull"
            tstar .= SmallestPointInConvexHull(sel)
        else
            # No need to use ConvexHull
            tstar .= sel[:,1]
        end
    end

    # Plot quantiles    
    for p in probs
        if p==0
            scatter!(pl, Tstar[1,[1]], Tstar[2,[1]], markercolor=:red, label="median")
        else
            r_index = argmin( abs.(p .- radii) )        
            true_p = radii[r_index]
            # @info true_p

            index = ((1+(r_index-1)*Ns+1):(1+r_index*Ns))
            index = [index; index[1]]
            
            plot!(pl,Tstar[1,index], Tstar[2,index], 
                    linestyle=:dash, 
                    linecolor=:red, 
                    legend=false; 
                    kwargs...)             
        end
    end
end

"""
    SmallestPointInConvexHull(points)
    
Find smallest point in the convex hull of the columns of `points`
""" 
function SmallestPointInConvexHull(points::Matrix{Float64})
    N = size(points,2)
    m = Model(Gurobi.Optimizer) 
    set_silent(m)

    @variable(m, w[1:N], lower_bound=0.0)    

    Q = points'points
    @objective(m, Min, w'Q*w )

    @constraint(m, total_weight, sum( w ) == 1.0 )
    optimize!(m)

    return points*value.(w)
end

# Superlevel set approach
""" 
    plotSLS(d::MixtureModel{Multivariate,Continuous,<:Distribution, <:DiscreteNonParametric}, 
        α::Vector{Float64}=[0.95]; 
        transf::Tuple{Function,Function}=(identity,identity), 
        nbins=150, 
        title, 
        fillarea=true, 
        c=cgrad([:black, :lightgray]),
        thickness::Tuple{Integer,Integer}=(1,1))

Create a plot of the superlevel set for mixture distribution `d` at levels `α` (vector)
"""
function plotSLS(d::MixtureModel{Multivariate,Continuous,<:Distribution, <:DiscreteNonParametric}, 
    α::Vector{Float64}=[0.95]; 
    transf::Tuple{Function,Function}=(identity,identity), 
    nbins=150, 
    title="$α SLS", 
    fillarea=true, 
    c=cgrad([:black, :lightgray]),
    thickness::Tuple{Integer,Integer}=(1,1))
    
    plotSLS!(plot(), d, α; transf, nbins, title, fillarea, c, thickness)        
end

""" 
    plotSLS!(d::MixtureModel{Multivariate,Continuous,<:Distribution, <:DiscreteNonParametric}, 
        α::0.95; 
        transf::Tuple{Function,Function}=(identity,identity), 
        nbins=150, 
        title, 
        fillarea=true, 
        c=cgrad([:black, :lightgray]),
        thickness::Tuple{Integer,Integer}=(1,1))

Add the superlevel set for mixture distribution `d` at scalar level `α` to an existing plot
"""
function plotSLS!(pl::Plots.Plot,
    d::MixtureModel{Multivariate,Continuous,<:Distribution, <:DiscreteNonParametric}, 
    α::Float64=0.95; 
    transf::Tuple{Function,Function}=(identity,identity), 
    nbins=150, 
    title="$α SLS", 
    fillarea=true, 
    c=cgrad([:black, :lightgray]),
    thickness::Tuple{Integer,Integer}=(1,1))
    
    plotSLS!(pl, d, [α]; transf, nbins, title, fillarea, c,thickness)
end

""" 
    plotSLS(d::MixtureModel{Multivariate,Continuous,<:Distribution, <:DiscreteNonParametric}, 
        α::0.95; 
        transf::Tuple{Function,Function}=(identity,identity), 
        nbins=150, 
        title, 
        fillarea=true, 
        c=cgrad([:black, :lightgray]),
        thickness::Tuple{Integer,Integer}=(1,1))

Create a plot of the superlevel set for mixture distribution `d` at scalar level `α`
"""
function plotSLS(d::MixtureModel{Multivariate,Continuous,<:Distribution, <:DiscreteNonParametric}, 
    α::Float64=0.95; 
    transf::Tuple{Function,Function}=(identity,identity), 
    nbins=150, 
    title="α SLS", 
    fillarea=true, 
    c=cgrad([:black, :lightgray]),
    thickness::Tuple{Integer,Integer}=(1,1))
    
    plotSLS(d, [α]; transf, nbins, title, fillarea, c, thickness)
end

"""
    plotSLS!(d::MixtureModel{Multivariate,Continuous,<:Distribution, <:DiscreteNonParametric}, 
        α::0.95; 
        transf::Tuple{Function,Function}=(identity,identity), 
        nbins=150, 
        title, 
        fillarea=true, 
        c=cgrad([:black, :lightgray]),
        thickness::Tuple{Integer,Integer}=(1,1))

Add the superlevel set for mixture distribution `d` at levels `α` (vector) to an existing plot

* `transf` is a tuple of functions with data transformation and inverse transformation. For example if log transformed data is used: transf = (log, exp)
* Note that the transformation needs to be strictly increasing and same transformation should be applied to all dimensions
"""
function plotSLS!(pl::Plots.Plot, 
    d::MixtureModel{Multivariate,Continuous,<:Distribution, <:DiscreteNonParametric}, 
    α::Vector{Float64}; 
    transf::Tuple{Function,Function}=(identity,identity), 
    nbins=150, 
    title="$α SLS", 
    fillarea=true, 
    c=cgrad([:black, :lightgray]),
    thickness::Tuple{Integer,Integer}=(1,1) )
    
    @assert length(d) == 2 "Dimension needs to be 2 for cdf"
    
    # Get transformations
    datatrans = transf[1]
    backtrans = transf[2]

    # Get scale of data
    limits = (0.001,0.999)
    if datatrans == log
        limits = (0.0001, 0.975)
    end
    (minx,maxx) = backtrans.(quantile.(marginal(d,1), limits))
    (miny,maxy) = backtrans.(quantile.(marginal(d,2), limits))

    # Create a grid
    rngx = range(minx, stop=maxx, length=nbins)
    rngy = range(miny, stop=maxy, length=nbins)
    
    # Get cdf at each grid point
    probs = [cdf(d, [datatrans(x); datatrans(y)]) for x in rngx, y in rngy]
        
    # Get prob of square
    probs[2:end,:] = probs[2:end,:] - probs[1:end-1,:]
    probs[:,2:end] = probs[:,2:end] - probs[:,1:end-1]
    
    # Indicators for whether a square is part of one of the SLS's
    insls::Matrix{Union{Float64,Missing}} = fill(missing, nbins, nbins)
        
    p = 0       # initially the area contains no probability
    @assert sum(probs)>=maximum(α) "SLS $(maximum(α)) not achievable, enlarge grid"
    
    for tp in sort(α, rev=false)        
    # Start adding squares, try to reach the smallest α value first
        while p < tp
            # Add the square with smallest prob, if there are ties, then add all
            i = findall(==(maximum(probs)), probs) 
            p = p + sum(probs[i])
            probs[i] .= 0
            insls[i] .= tp
        end                
    end    
    
    # Create plots (need to shift rng for proper heatmap axes)
    if fillarea==true
        heatmap!(pl, rngx .- (rngx[2]-rngx[1])/2, rngy .- (rngy[2]-rngy[1])/2, insls', title=title, c=c)
    else
        heatmap!(pl, rngx .- (rngx[2]-rngx[1])/2, rngy .- (rngy[2]-rngy[1])/2, reducetocontour(insls',thickness), title=title, c=c)
    end
    
    # Perform some performance checks and obtain plot limits
    counts = [sum(skipmissing(c)) for c in eachrow(insls)]        
    highrow = findlast(>(0), counts)
    if highrow == length(counts)
        @warn "Increase grid or consider extending rhs of x-range in search for SLS"
    end
    highrow = rngx[min(highrow+1, length(counts))]
    
    lowrow = findfirst(>(0), counts)
    if lowrow == 1
        @warn "Increase grid or consider extending lhs of x-range in search for SLS"
    end
    lowrow = rngx[max(lowrow-1, 1)]    

    counts = [sum(skipmissing(c)) for c in eachcol(insls)]    
    highcol = findlast(>(0), counts)
    if highcol == length(counts)
        @warn "Increase grid or consider extending rhs of y-range in search for SLS"
    end
    highcol = rngy[min(highcol+1, length(counts))]
    
    lowcol = findfirst(>(0), counts)
    if lowcol == 1
        @warn "Increase grid or consider extending lhs of y-range in search for SLS"
    end
    lowcol = rngy[max(lowcol-1, 1)]
    
    # Adjust plot limits
    xlims!(pl, lowrow, highrow)
    ylims!(pl, lowcol, highcol)

    return pl
end

"""
    reducetocontour(realinput::AbstractMatrix{Union{Missing,T}}, thickness::Tuple{Integer,Integer}) where T<:Number

Reduce an SLS to its contour, `thickness`` is a tuple indicating the number of squares on the x and y axis to be used in the contour
"""
function reducetocontour(realinput::AbstractMatrix{Union{Missing,T}}, thickness::Tuple{Integer,Integer}) where T<:Number
    
    # Process input by row, whiteout squares if they are not part of the contour
    function internal(realinput::AbstractMatrix{Union{Missing,T}}, thickness::Integer) where T<:Number                
        out   = copy(realinput)
        input = copy(realinput)
        input[ismissing.(realinput)] .= 1
        
        for (rin, rout) in zip(eachrow(input), eachrow(out))            
            a = firstindex(rin)                      
            while a <= lastindex(rin)
                valfound = rin[a]
                if a != firstindex(rin)
                    @assert rin[a]!=rin[a-1]
                end
                # find where the value will be different, at `b` a different value applies                
                b = findnext(!=(valfound), rin, a+1)
                                
                if b==nothing                
                # change cannot be found, so area ends at end of vector                
                    offset = 1
                    if (a>firstindex(rin)) && (rin[a-1] < rin[a])
                        offset = 0
                    end
                    rout[(a+offset):(size(input,2)-1)] .= missing               
                    a = size(input,2)+1
                else          
                # transition to a new area                            
                    offsetend = 1 
                    if rin[b] > rin[b-1]
                    # area to the right has higher value (larger SLS area), 
                        offsetend = 1+thickness
                    end
                  
                    offset = 1 + thickness-1
                    if (a>firstindex(rin)) && (rin[a-1] < rin[a])
                    # not outer most interval and moving to larger area
                        offset = 0                        
                    end
                    s = max(firstindex(rout),a+offset)
                    e = b-offsetend
                    rout[s:e] .= missing
                    a = b
                end
                
            end
        end
        out[findall(==(1), skipmissing(out))] .= missing
        return out
    end
    
    A = internal(realinput, thickness[1])    
    B = internal(realinput', thickness[2])'    
    A[ismissing.(A)] .= B[ismissing.(A)]   
    
    return A
end

end