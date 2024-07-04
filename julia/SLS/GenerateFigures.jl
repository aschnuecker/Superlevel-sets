using Pkg
Pkg.activate(".")
Pkg.instantiate()       # Download all packages to the required version (if needed)

import InteractiveUtils:versioninfo
import Dates:now

using LaTeXStrings
using Random
using LinearAlgebra
using Distributions
using Plots
using ImplicitEquations
using MAT

include("SLS.jl")
using .SLS

function Generate_Figure1()    
    
    println("""
    
    Recreating Figure 1...
    This requires a valid licence for the Gurobi solver.
    The execution takes quite some time, and uses a lot of memory to 
    solve the Optimal Transport problem
    """)
    
    N    = 100_000  # Number of points used in the creation of the contours
    np   = 75       # Parameter in the Center Outward method (np^2+1 data points are used)
    prob = 0.8      # Desired probability content
    
    # First sub-figure    
    m =  MixtureModel([ MvNormal([-3, 0], 0.5*I),
                        MvNormal([+3, 0], 0.5*I)
                      ], [1,1]/2)
    
    # Generate data
    Random.seed!(2022)
    data = rand(m, N)
           
    # Directional Koenker-Bassett quantile
    p2 = DirectionalKB(m, data', prob, 0.1π)
    plot!(p2, aspect_ratio=:equal, title="", xlabel=L"y_1", ylabel=L"y_2")
    
    # Elliptical quantile
    q = ellipsoid_quantile(data, prob)
    plot!(p2, Eq(q[1], q[2]), xlims=extrema(data[1,:]), ylims=extrema(data[2,:]).*1.15, label="")
    plot!(p2,[],[], linecolor=:black, label="Elliptical")
    
    # Center-Outward quantile
    Random.seed!(2022)   
    data = rand(m, np*np+1)        
    CenterOutward!(p2, data, Nr=np, Ns=np, probs=prob, label="Center-outward")
        
    # Second sub-figure
    m2 =  MixtureModel([MvNormal([ 0,   0],    cor2var(-0.3, [0.2, 0.2])),
                        MvNormal([-0.8, 1.8],  cor2var(-0.3, [0.4, 0.4])), 
                        MvNormal([ 4.8, 1.8],  cor2var(0.3,  [0.4, 0.4])),
                        MvNormal([ 2,  -1],    [0.75, 0.2]),
                        MvNormal([ 4,   0],    cor2var(0.3,  [0.2, 0.2]))                        
                        ], [1,1,1,1,1]/5)

    # Generate data
    Random.seed!(2022)
    data2 = rand(m2, N)    

    # Directional Koenker-Bassett quantile
    p4 = DirectionalKB(m2, data2', prob, 0.1π)
    plot!(p4, aspect_ratio=:equal, title="", xlabel=L"y_1", ylabel=L"y_2")
    
    # Elliptical quantile
    q = ellipsoid_quantile(data2, prob)
    plot!(p4, Eq(q[1], q[2]), xlims = extrema(data2[1,:]), ylims=extrema(data2[2,:]), label="")    
    plot!(p4, [], [], linecolor=:black, label="Elliptical")

    # Center-Outward quantile
    Random.seed!(2022)       
    data = rand(m2, np*np+1)        
    CenterOutward!(p4, data, Nr=np, Ns=np, probs=prob, label="Center-outward")
    plot!(p4)
    
    # Combining both plots and save to file
    plot(p2, p4, legend = :outertopright, legendfontsize=6) |> display
    file = "plots/Figure1_$(now()).pdf"
    println("Saving Figure 1 as $file")
    savefig(file)    
end

function Generate_Figure2()    
# Generates the Superlevel-sets [SLSs] for two example distributions
    
    # Example 1
    Random.seed!(2022)
    m =  MixtureModel([ MvNormal([-3, 0], 0.5*I),
                        MvNormal([+3, 0], 0.5*I)
                        ], [1,1]/2)

    data = rand(m, 100_000)    
    p1 = scatter( data[1,1:min(1000,size(data,2))], 
                  data[2,1:min(1000,size(data,2))], 
                  label="",
                  markerstrokewidth = .2)  
    
    plotSLS!(p1, m, 0.8, nbins=250, 
                         fillarea=false,
                         c=cgrad([:black,:black]),
                         thickness=(2,3)) 
    
    # Example 2
    Random.seed!(2022)   
    m2 =  MixtureModel([MvNormal([ 0,   0],   cor2var(-0.3, [0.2, 0.2])),
                        MvNormal([-0.8, 1.8], cor2var(-0.3, [0.4, 0.4])), 
                        MvNormal([ 4.8, 1.8], cor2var(0.3,  [0.4, 0.4])),
                        MvNormal([ 2,  -1],   [0.75, 0.2]),
                        MvNormal([ 4,   0],   cor2var(0.3,  [0.2, 0.2]))                        
                    ], [1,1,1,1,1]/5)
    data2 = rand(m2, 100_000)        
    p2 = scatter( data2[1,1:min(1000,size(data2,2))], 
                  data2[2,1:min(1000,size(data2,2))],
                  label="",
                  markerstrokewidth = .2) 
    plotSLS!(p2, m2, 0.8, nbins=250, 
                          fillarea=false,
                          c=cgrad([:black,:black]),
                          thickness=(2,3)) 
    
    # Combine both examples in single plot and save
    plot!(p1, aspect_ratio=:equal, title="", xlabel=L"y_1", ylabel=L"y_2")
    plot!(p1, xlims = extrema(data[1,:]), ylims=extrema(data[2,:]).*1.15)
    plot!(p1, colorbar=:none)

    plot!(p2, aspect_ratio=:equal, title="", xlabel=L"y_1", ylabel=L"y_2")
    plot!(p2, xlims = extrema(data2[1,:]), ylims=extrema(data2[2,:]))
    plot!(p2, colorbar=:none)

    plot(p1,p2) |> display
    file = "plots/Figure2_$(now()).pdf"
    println("Saving Figure 2 as $file")
    savefig(file)
end
    
function Generate_Figure6()
# Obtain the SLS figures for the empirical application
    ind  = [2,3]
    prob = [0.2, 0.4, 0.6, 0.8]
    thickness = (2, 2)    

    @assert findfirst(==(1), ind) == nothing "Cannot include variable 1 in input vector"
    
    nms = ["income", "food", "housing", "utility"]    
    transf = (log, exp)
            
    # Load posterior estimates 
    matfile = matopen("../../Matlab/inputJulia.mat")

    meanData = read(matfile, "meanlogdata")[:]
    stdData  = diagm(read(matfile, "stdlogdata")[:])
    Σ = read(matfile, "Sigma_m_post")
    μ = read(matfile, "mu_m_post")                   

    for (m,s) in zip(eachcol(μ), eachslice(Σ,dims=3))
        s .= Symmetric(stdData*s*stdData)
        m .= stdData*m .+ meanData
    end

    w = read(matfile, "kappa_post")[:]
    w = w / sum(w)
    d = MixtureModel(MvNormal.(eachcol(μ),eachslice(Σ, dims=3)), w)

    # Helper function to decorate plot
    function decorateplot!(p)
        scale = 1000
        plot!(p, colorbar=:none)
        if transf==identity
            plot!(p, xlabel="log "*nms[ind[1]], ylabel="log "*nms[ind[2]])
        else
            plot!(p, xlabel=nms[ind[1]]*"(x$scale)", ylabel=nms[ind[2]]*"(x$scale)")
        end
        plot!(p, titlefont=10, formatter = y -> round(Int, y / scale))
        plot!(p, xlabelfontsize=10, ylabelfontsize=10)
    end
        
    # Create plot
    pls = []
    condOn = 1
    d2 = marginal(d,[condOn, ind...])           # drop variables that are not used and move conditiong variable to position 1
    
    for q in prob
        val = quantile(marginal(d,condOn), q)   # Value to condition on
        
        p2 = plotSLS(conditional(d2, 1, val), [0.2, 0.4, 0.6, 0.8], 
                nbins=350, 
                transf=transf,
                title="$(nms[condOn])=Q($(q))",
                fillarea=false,
                c=cgrad([:blue,:blue]),
                thickness=thickness)
        decorateplot!(p2)

        # drop labels for some of the plots
        if length(prob)==4
            if q==prob[1] || q==prob[2]
                plot!(p2,xlabel="")
            end
            if q==prob[2] || q==prob[4]
                plot!(p2,ylabel="")
            end
        end
        plot!(p2)
        
        push!(pls,p2)       # Add plot to list of plots
    end

    # Add SLS after 6000 income increase
    for (i,q) in enumerate(prob)
        val = quantile(marginal(d,condOn), q)
        @info ("income $(100*q)%", val, exp(val))
        
        plotSLS!(pls[i], conditional(d2, 1, log(exp(val)+6000)), [0.2, 0.4, 0.6, 0.8], 
                nbins=350, 
                transf=transf,
                title="$(nms[condOn])=Q($(q))(+6,000)",
                fillarea=false,
                c=cgrad([:red,:red]),
                thickness=thickness)
    end        
    
    # Scale all graphs equally
    minX = minimum(xlims(p)[1] for p in pls)
    maxX = maximum(xlims(p)[2] for p in pls)

    minY = minimum(ylims(p)[1] for p in pls)
    maxY = maximum(ylims(p)[2] for p in pls)

    # Add title to each subplot and adjust ticks
    for p in pls
        xlims!(p, minX,maxX)
        ylims!(p, minY,maxY)
        
        annotate!(p, (minX+maxX)/2, maxY, text(p.subplots[1].attr[:title], :center, 7))
        plot!(p, title="")
        plot!(p, yticks=0:10000:maxY)
        plot!(p, xticks=0:5000:maxX)
    end
    
    str = ""
    for p in prob
        str = str*"$p, "
    end
    str = str[1:length(str)-2]
    
    p = plot(pls..., plot_title="Bivariate $str superlevel sets conditional on income", plot_titlevspan=0.06, plot_titlefont=10) |> display
    
    file = "plots/Figure6_$(now()).pdf"
    println("Saving Figure 6 as $file")
    savefig(file)    
end
    
function main()
    versioninfo()

    println("""
    
    Recreating Figures 1, 2 and 6 of
        On superlevel sets of conditional densities and multivariate quantile regression,
        Journal of Econometrics
        Camehl, Fok, and Gruber    
    """)

    Generate_Figure2()
    Generate_Figure6()
    Generate_Figure1() # takes a long time
end

main()