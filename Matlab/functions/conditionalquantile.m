function [q_mix] = conditionalquantile(idx_eq,idx_ex,t_start,t_end,n_q,M,q_grid,K,mu,Sigma,q_marginal,kappa)
epsilon        = 0.00001;

Tleng = length(t_start:t_end);
% ------------- output matrices -------------------------------------------
q_mix          = zeros(Tleng,n_q);
sigma_c        = zeros(K,M);
mu_c           = zeros(t_end,K,M);
q_c            = zeros(t_end,K,M,n_q);
w_m            = zeros(M,t_end,K,n_q);
w              = zeros(M,t_end,K);
% -------------------------------------------------------------------------

for m = 1:M
   % --- Step 2a: conditional moments per component for each equation
   temp_cov       = squeeze(Sigma(idx_eq,idx_ex,m));
   temp_siginv    = inv(squeeze(Sigma(idx_ex,idx_ex,m)));

   % --- conditional quantile function per component for each equation
   for t = t_start:t_end
      % conditional mean and variance
       mu_c(t,idx_eq,m)  =  mu(idx_eq,m) + (q_marginal(t,:)- mu(idx_ex,m)')*temp_siginv*temp_cov';
       sigma_c(idx_eq,m) =  Sigma(idx_eq,idx_eq,m) - temp_cov*temp_siginv*temp_cov';

       % for each q
       for idx_q = 1:length(q_grid)
            q = q_grid(idx_q);
            % conditional quantile for each component
            q_c(t,idx_eq,m,idx_q) = norminv(q,mu_c(t,idx_eq,m),sqrt(sigma_c(idx_eq,m)));
            % conditional weights 
            w_m(m,t,idx_eq,idx_q) = kappa(m)*mvnpdf(q_marginal(t,:),mu(idx_ex,m)'.*ones(1,length(idx_ex)),Sigma(idx_ex,idx_ex,m));
       end                   
   end
end

% --- Step 2b: bisection method  
% --- min and max of conditional quantile per equation
t_idx = 1;
for t = t_start:t_end
    % for each q
    for idx_q = 1:length(q_grid)
        q = q_grid(idx_q);
        if M == 1
            % if only one component, no need for bisection
            q_mix(t_idx,idx_q)   = norminv(q,mu_c(t,idx_eq,1),sqrt(sigma_c(idx_eq,1)));                       
        else
            % --- conditional weights (for equation idx_eq)
            w(:,t,idx_eq) = w_m(:,t,idx_eq,idx_q)./sum(w_m(:,t,idx_eq,idx_q));   

            q_c_min(idx_eq) = min(q_c(t,idx_eq,:,idx_q));
            q_c_max(idx_eq) = max(q_c(t,idx_eq,:,idx_q));

            f = @(z) mixQ(q,z,w(:,t,idx_eq),mu_c(t,idx_eq,:),sigma_c(idx_eq,:));
            q_mix(t_idx,idx_q) = bisection(f,q_c_min(idx_eq)-epsilon,q_c_max(idx_eq)+epsilon,0);
        end
    end
    t_idx = t_idx+1;
end 