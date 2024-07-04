
function [q_mix,q_fix] = calculationquantilesim(q_grid,K,M,T,T_q,T_fix,n_fix,n_q,y,Y_temp,mu,Sigma,kappa)

% Quantile estimation for simulation
epsilon    = 0.00001;

% ------------- output matrices -------------------------------------------
q_mix      = zeros(T_q+1,K,n_q);
q_fix      = zeros(T_fix,length(n_fix),n_q);

% -------------------------------------------------------------------------

%% Conditional quantile 

for idx_eq = 1:K
   idx_ex             = 1:K;
   idx_ex(idx_eq)     = [];
    q_mix(:,idx_eq,:) = conditionalquantile(idx_eq,idx_ex,(T-T_q),T,n_q,M,q_grid,K,mu,Sigma,y(:,idx_ex),kappa);
end

%% Conditional quantiles with conditional variables with fixed values
% conditional quantile for y_k with fixed y_{-k}
idx_eq           = 1;   
idx_ex           = 1:K;
idx_ex(idx_eq)   = [];
for j = 1:length(n_fix)
    Y_tempall    = [Y_temp(:,j) ones(T_fix,1)*n_fix(j)];
    q_fix(:,j,:) = conditionalquantile(idx_eq,idx_ex,1,T_fix,n_q,M,q_grid,K,mu,Sigma,Y_tempall,kappa);
end

end