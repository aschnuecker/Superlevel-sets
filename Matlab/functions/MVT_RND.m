function sample = MVT_RND(mu, cov_mat, df, ndraws)

% Purpose: 
% Generate random numbers from multivariate t distribution
% -----------------------------------
% Density:
% f(x) = constant * det(cov_mat)^(-1/2) * (1+(x-mu)'*inv(cov_mat)*(x-mu)/df)^(-(df+k)/2) 
% where k is the dimension of MVT
% E(X) = mu, Cov(X) = cov_mat * df/(df-2)
% -----------------------------------
% Algorithm: 
% Use Matlab canned mvtrnd function, but repacked with covariance matrix
% -----------------------------------
% Usage:
% mu = mean vector of multivariate t distribution (k*1)
% cov_mat = covariance matrix of multivariate t distribution (k*k)
% df = degree of freedom
% ndraws = number of draws
% -----------------------------------
% Returns:
% sample = random numbers from MVN(mu,cov_mat), k * ndraws matrix
% -----------------------------------
% Notes:
% Support matrix input of mu, which should be k * ndraws
% In that case, no need to specify ndraws
% It will return a matrix of random numbers.
%
% Written by Hang Qian, Iowa State University
% Contact me:  matlabist@gmail.com

dim = size(cov_mat, 1);
[nrow,ncol] = size(mu);
if nrow ~= dim && ncol == dim  
    mu = mu';
    [nrow,ncol] = size(mu);
end

if nargin < 4 
    ndraws = max(ncol , length(df));
end


rescale = sqrt(diag(cov_mat));
corr_mat = cov_mat ./ (rescale * rescale');

if ndraws > 1
    if ncol == 1 && nrow > 1
        mu = mu(:,ones(ndraws,1));
    end
    rescale = rescale(:,ones(ndraws,1));
end

sample = mu +  mvtrnd(corr_mat, df,ndraws)' .* rescale;