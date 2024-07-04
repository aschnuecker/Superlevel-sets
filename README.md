# Replication codes of *On superlevel sets of conditional densities and multivariate quantile regression*

Annika Camehl (camehl@ese.eur.nl), Dennis Fok (dfok@ese.eur.nl), and Kathrin Gruber (gruber@ese.eur.nl)

Econometric Institute, Erasmus University Rotterdam, Burgemeester Oudlaan 50
3062 PA Rotterdam, The Netherlands

July 2024

## 1. Overview

The code in this replication package replicates the results shown in Sections 2, 5 \& 6 of the paper:

Camehl, Fok, Gruber, 2024, *On superlevel sets of conditional densities and multivariate quantile regression*, Journal of Econometrics, https://doi.org/10.1016/j.jeconom.2024.105807.

All data sources are listed in [Section 2](#2-data). The code is run in Matlab and Julia. 

## 2. Data
The data are a subset of households from the U.S. Consumer Expenditure Survey collected by the Bureau of Labor Statistics in 2015. The full survey interview files are public available via https://www.bls.gov/cex/pumd_data.htm. The analysis in Section 6 is based on the variable labels: `houspq` (housing), `foodpq` (food), `utilpq` (utilities) and `fincbtxm` (income before taxes). The data are stored in the file `CEX_2015_b.csv`.


## 3. Code

### Software requirements
The code is build on Matlab R2023a (https://matlab.mathworks.com/) and Julia 1.10.3 (https://julialang.org). We use the Parallel Computing Toolbox in Matlab.

### Description of Matlab programs

**Main files**

The Matlab files appear in the `Matlab` subfolder.
The file `QR_Simulation.m` replicates the result in Section 5 *Simulation Study*. The file `QR_Application.m` replicates the result in Section 6 *Empirical Application*.

**Functions and additional files**

* `bisection.m` implements bi-section needed in the calculation of univariate conditional quantiles, step 7 in Algorithm 2 of the paper. Source: http://www.mathworks.com/matlabcentral/fileexchange/28150.
* `calculationquantile.m` calculates marginal univariate quantiles and conditional univariate quantiles for the application by calling `conditionalquantile.m`.
* `calculationquantilesim.m` calculates conditional univariate quantiles based on simulated or fixed values for the conditional variables for the simulation by calling `conditionalquantile.m`.
* `conditionalquantile.m` function to calculate the conditional univariate quantiles based on Algorithm 2 of the paper. Quantiles can be calculated for each simulation draw or based on posterior mean estimates.
* `gigrnd.m` implements the Devroye (2014) algorithm for sampling from the generalized inverse Gaussian (GIG) distribution. Source: https://www.mathworks.com/matlabcentral/fileexchange/53594-gigrnd-p-a-b-samplesize.
* `mixQ.m` gives input to the bi-section method based on the Gaussian mixture model.
* `MVT_RND` generates random numbers from a multivariate t-distribution (needed in the simulation for the DGP which follows a multivariate t-distribution). Source: https://www.mathworks.com/matlabcentral/fileexchange/32601-toolkit-on-econometrics-and-economics-teaching?s_tid=prof_contriblnk.
* `quantiletrue.m` calculates the population quantiles of the data generating processes in Section 5 *Simulation Study*. The population quantile functions are given in the Supplementary Appendix of the paper.
* `rq_fnm.m` solves a linear program by the interior point method needed in the calculation of the comparison models to calculate standard univariate quantiles used in Section 5 *Simulation Study*.  Source: https://nl.mathworks.com/matlabcentral/fileexchange/70072-quantile-regression.
* `sc_postprocessing.m` implements the post-processing step to solve the label switching problem if case component-specific posterior estimates are needed. The steps are discussed in Section 3.2 *Posterior Inference*.
* `sc_prior.m` sets the prior distributions as discussed in Section 3.1 *Prior Distributions*.
* `sc_sampler.m` runs the posterior sampler discussed in Section 3.2 *Posterior Inference* and Appendix B *Posterior Simulation Algorithm*.
* `shadeplot.m` draws two lines on a plot and shades the area between those lines, used for Figure 7.
* `simdgp.m` simulates data from the data generating processes given in Section 5 *Simulation Study*: multivariate Gaussian (*DGP=1*), multivariate Student-t (*DGP=6*), multivariate log-Gaussian (*DGP=2*), conditional heteroskedasticity (*DGP=7*), and multivariate Gaussian mixture (*DGP=3*).
* `univariatequantile.m` calculates three types of standard univariate quantile regression models for comparison: (1) a standard univariate quantile regression model with the linear regression quantiles estimated independently
for each response variable, (2) a univariate linear quantile regression model with the other response variables of the output-vector in the conditioning set, and (3) a univariate non-linear quantile regression model with the level and squared response variables in the conditioning set.


### Description of Julia programs
The Julia files appear in the `julia` subfolder. This folder contains
* `Manifest.toml` and `Project.toml`: files specifying the used packages (and the exact version of each package);
* `SLS/GenerateFigures.jl`: main file to recreate the results;
* `SLS/SLS.jl`: code for calculating the superlevel-sets and various multivariate quantiles;
* `SLS/plots/`: directory in which the generated plots are stored in pdf format. Filenames contain a date stamp.

All functions are documented in the corresponding file.

## 4. Instructions to Replicators

### Simulation: `QR_Simulation.m`

The simulation code produces the results for one specified DGP. At the beginning of the code you need to set the variable `DGP`: multivariate Gaussian (*DGP=1*), multivariate Student-t (*DGP=6*), multivariate log-Gaussian (*DGP=2*), conditional heteroskedasticity (*DGP=7*), and multivariate Gaussian mixture (*DGP=3*). `nMC` sets the total number of simulation draws (we used `nMC=1000` in the paper). Running 1000 simulation draws for one DGP takes considerable time (roughly 30 hours on a cluster computer using 16 nodes).

To produce Figures 3, 4, as well as Supplementary Appendix Figures 1, 2, and 3 and Table 2, `QR_Simulation.m` needs to be run for each DGP. That is, to produce Figure 3 set `DGP=3`, Figure 4 set `DGP=7`, Supplementary Appendix Figure 1 set `DGP=1`, Supplementary Appendix Figure 2 set `DGP=6`, and Supplementary Appendix Figure 3 set `DGP=2`. `QR_Simulation.m` produces the column of Table 2 for the set DGP.

### Empirical Application: `QR_Application.m`
The file produces the output file `inputJulia.mat` which is used in the Julia code as input. 

Note that if you want to calculate quantiles based on the posterior means, you also need the post-processing step (to deal with label switching). To do so, set *postprocess=1* in line 24 of the file `QR_Application.m`.

### Creation of SLS figures: `GenerateFigures.jl`
This code to recreates Figures 1, 2 and 6 of the above mentioned paper. The code is written in Julia  and is last tested for Julia Version 1.10.3.

To execute the code
* Install julia version 1.10.3 ([juliaup](https://github.com/JuliaLang/juliaup) is recommended to easily manage multiple versions for julia)	
* Open a terminal and set the working directory to `[repository]/julia/SLS/`
* Run the code using `julia --threads=auto GenerateFigures.jl`
* The plots can be found in the subdirectory named `plots`

Alternatively one can use (for example) VSCode and execute the different functions from there. Each figure is generated by a separate function in `GenerateFigures.jl`.


**Notes**
* The code should also work with newer versions of Julia;
* The random number generator in Julia can generate different random numbers across Julia versions (even when fixing the seed). The random samples generated are therefore not the same as in the published paper. The results in the paper were generated with Julia 1.8.2;
* Figure 6 relies on posterior means obtained using MCMC estimation. This part of the calculation is done in Matlab, see the other files in this repository. Matlab saves the posterior means to `inputJulia.mat`, in this part of the code they are read from this file;
* The Center-Outward method to obtain multivariate quantiles relies on solving a (large) Optimal Transportation problem. We rely on the Gurobi solver v1.0.2. This solver requires a separate licence. Without such a license Figure 1 cannot be recreated. Trial licenses or academic licenses are available (https://www.gurobi.com);
* (Our approach to) the Center-Outward method is also rather slow (±40 minutes for each instance on a i7-1260P with 32GB memory) and consumes quite some memory. The solver may run out of memory and get killed. One may want to reduce the value of `np` on line 30 of `SLS/GenerateFigures.jl`.


# 5. List of figures and tables

| Figure/Table | Program file | Lines |  
|----------|-------|  
|Figure 1 | `GenerateFigures.jl` | 19--88 | 
|Figure 2 | `GenerateFigures.jl`| 90--141|  
|Table 2 | `QR_Simulation.m` | 320--327|  
|Figure 3 | `QR_Simulation.m`| 257--318|  
|Figure 4 | `QR_Simulation.m`| 257--318|  
|Figure 5 | `QR_Application.m`| 47--75|  
|Figure 6 | `GenerateFigures.jl`| 143--258|  
|Figure 7 | `QR_Application.m`| 199--249|  
|Figure 1- Supplementary Appendix | `QR_Simulation.m`| 257--318|  
|Figure 2- Supplementary Appendix | `QR_Simulation.m`| 257--318|  
|Figure 3- Supplementary Appendix | `QR_Simulation.m`| 257--318|  
