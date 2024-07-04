function [q_marginal,q_mix,q_mix_mar] = calculationquantile(stdlogdata,meanlogdata,q_grid,K,M,Tfix,n_q,mu_m,Sigma_m,kappa,delta)


epsilon        = 0.00001;

% ------------ re-do standardization of coefficients ----------------------
mu             = repmat(stdlogdata',1,5).*mu_m + repmat(meanlogdata',1,5);
Sigma          = zeros(K,K,M);
for m = 1:M
  Sigma(:,:,m) = diag(stdlogdata)*squeeze(Sigma_m(:,:,m))*diag(stdlogdata);
end
% ------------- output matrices -------------------------------------------
q_mix          = zeros(Tfix,K-1,n_q);
q_mix_mar      = zeros(Tfix,K-1,n_q);
q_true_mix     = zeros(1,M);
q_marginal     = zeros(Tfix,1);
% -------------------------------------------------------------------------

%% --- (1) MARGINAL QUANTILE: calculate marginal quantile for income (y1)

% --- marginal quantiles
idx_mar         = 1;
q_grid_fix      = linspace(0.1,0.9,Tfix);
 for idx_q = 1:Tfix
    q                 = q_grid_fix(idx_q); 
    for m = 1:M
       mu_c_m(m)     =  mu(idx_mar,m);
       sigma_c_m(m)  =  Sigma(idx_mar,idx_mar,m);
       q_true_mix(m) =  norminv(q,mu_c_m(m),sqrt(sigma_c_m(m)));
    end
    q_c_min = min(q_true_mix);
    q_c_max = max(q_true_mix);

    % bisection method to calculate quantile of mixture
    f                 = @(z) mixQ(q,z,kappa,mu_c_m,sigma_c_m);
    q_marginal(idx_q) = bisection(f,q_c_min-epsilon,q_c_max+epsilon,0);  
end

%% --- (2) CONDITIONAL QUANTILE: calculate bi-variate conditional quantiles for all variables conditional on income (y1)

for idx_eq = 2:K
   idx_ex = 1;
   q_mix(:,idx_eq-1,:) = conditionalquantile(idx_eq,idx_ex,1,Tfix,n_q,M,q_grid,K,mu,Sigma,q_marginal,kappa);
end

%% --- (3) MARGINAL EFFECTS
y_new = log(exp(q_marginal) + delta);
for idx_eq = 2:K
    idx_ex = 1;
   q_mix_mar(:,idx_eq-1,:) = conditionalquantile(idx_eq,idx_ex,1,Tfix,n_q,M,q_grid,K,mu,Sigma,y_new,kappa);
end



end