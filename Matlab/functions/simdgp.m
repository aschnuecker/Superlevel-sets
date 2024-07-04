function [Y, q_true,  q_true_fix, q_marg, q_cond] = simdgp(gendata,gentrueq,gentrueqfix,marginalq,condq,DGP,T,K,q,n_fix,Y_temp1,y_MC,idx_qfix,idx_mar)
%--------------------------------------------------------------------------
% simulate DGP
%--------------------------------------------------------------------------
% INPUTS:
% --- DGP
% DGP=1: u~Normal; 
% DGP=2: u~lognorm; 
% DGP=3: u~3-component mixture; 
% DGP=6: u~multi-variate t-distribution;
% DGP=7: conditional heteroskedasticity;

% Covariance matrix
Sigma_U = 0.25*ones(K,K)+0.15*eye(K);

epsilon = 0.0001;

if DGP == 1 
    mu = 0.2*ones(K,1);
elseif DGP == 2
    mu = 0.2*ones(K,1);
elseif DGP == 3
    H   = zeros(T,3);
    Sigma_M = zeros(K,K,3);
    Sigma_M(:,:,1) = Sigma_U;
    Sigma_M(:,:,2) = eye(K);
    Sigma_M(:,:,3) = 0.2*eye(K) + 0.5*ones(K,K);
    mu_M  = [2 0 -2; 2 0 0.5; 2 0 1];
    kappa = [1/3 1/3 1/3]';

elseif DGP == 6
    mu = 0.2*ones(K,1);
    df = 5;
elseif DGP == 7 
    d_mix = 1000; 

    mu    = [0.2 0.2 0.2]';
    kappa = (1/d_mix)*ones(d_mix,1);     
    Omega = zeros(K,K,d_mix);
   for idx_d = 1:d_mix
       z                = normrnd(0,1);
       Omega(:,:,idx_d) = exp(z)*Sigma_U;

   end
end

T_fix = size(Y_temp1,1);

%----------------------GENERATE--------------------------------------------
q_true      = zeros(T,K);
q_true_fix  = zeros(T_fix,length(n_fix));
q_marg      = [];
q_cond      = [];
% --- generate Y from Multiple output regression model

% ---------------------------------------------------------------------
% ----- DGP1
if DGP == 1
    % --- normal distribution 
    for t = 1:T
        if gendata == 1
            u  = mvnrnd(mu,Sigma_U,1);
            % generate data
            y_MC(t,:) = u;
        end

        % true quantile
        if gentrueq == 1
            for idx = 1:K
               q_true(t,idx) = quantiletrue(DGP,q,idx,Sigma_U,mu,y_MC(t,:),K,0,0);
            end
        end
    end
    if gentrueqfix == 1
        for t = 1:T_fix
            for j = 1:length(n_fix) 
               Y_temp      = [Y_temp1(t,j) n_fix(j)]; % set fixed value for other outcome variables    
               q_true_fix(t,j) = quantiletrue(DGP,q,idx_qfix,Sigma_U,mu,Y_temp,K,0,1);
            end
        end
    end
   if marginalq == 1
      q_marg = norminv(q,mu(idx_mar),sqrt(Sigma_U(idx_mar,idx_mar)));
   end
   if condq == 1
       q_cond      = zeros(n_fix,1);
       for j = 1:n_fix 
            q_cond(j) = quantiletrue(DGP,q,1,Sigma_U(idx_qfix:K,idx_qfix:K),mu(idx_qfix:K),Y_temp1(j),K-1,0,1);
       end
   end
% ---------------------------------------------------------------------
% ----- DGP2                 
elseif DGP == 2 
    % --- log normal  
    for t = 1:T
        if gendata == 1
            y_MC(t,:)  = exp(mvnrnd(mu,Sigma_U,1));
        end

        % true quantile
        if gentrueq == 1
            for idx = 1:K
               q_true(t,idx) = quantiletrue(DGP,q,idx,Sigma_U,mu,y_MC(t,:),K,0,0);
            end
        end
    end
    
    if gentrueqfix == 1
        for t = 1:T_fix
           for j = 1:length(n_fix) 
               Y_temp      = [Y_temp1(t,j) n_fix(j)];
               q_true_fix(t,j) = quantiletrue(DGP,q,idx_qfix,Sigma_U,mu,Y_temp,K,0,1);
           end   
        end
    end
    if marginalq == 1
      q_marg = exp(norminv(q,mu(idx_mar),sqrt(Sigma_U(idx_mar,idx_mar))));
    end
    if condq == 1
       q_cond      = zeros(n_fix,1);
       for j = 1:n_fix 
            q_cond(j) = quantiletrue(DGP,q,1,Sigma_U(idx_qfix:K,idx_qfix:K),mu(idx_qfix:K),Y_temp1(j),K-1,0,1);
       end
   end
% ---------------------------------------------------------------------
% ----- DGP3    
elseif DGP == 3 
    % --- mixture: 3 component, normal
    for t = 1:T
        % generate data
        if gendata == 1
            H(t,:) = mnrnd(1,kappa);
            if H(t,1) == 1
               u         = mvnrnd(mu_M(:,1),Sigma_M(:,:,1),1);
               y_MC(t,:) = u;
            elseif H(t,2) == 1
               u         = mvnrnd(mu_M(:,2),Sigma_M(:,:,2),1);
               y_MC(t,:) = u;
            elseif H(t,3) == 1
               u         = mvnrnd(mu_M(:,3),Sigma_M(:,:,3),1);
               y_MC(t,:) = u;
            end
        end
        % true quantile
        if gentrueq == 1
            for idx = 1:K
                q_true(t,idx) = quantiletrue(DGP,q,idx,Sigma_M,mu_M,y_MC(t,:),K,kappa,0);
            end
        end
    end
    
    if gentrueqfix == 1
        for t = 1:T_fix
           for j = 1:length(n_fix) 
               Y_temp      = [Y_temp1(t,j) n_fix(j)];
               q_true_fix(t,j) = quantiletrue(DGP,q,idx_qfix,Sigma_M,mu_M,Y_temp,K,kappa,1);
           end           
        end
    end
    if marginalq == 1
       for m = 1:3
           mu_c_m(m)     =  mu_M(idx_mar,m);
           sigma_c_m(m)  =  Sigma_M(idx_mar,idx_mar,m);
           q_true_mix(m) =  norminv(q,mu_c_m(m),sqrt(sigma_c_m(m)));
       end
        q_c_min = min(q_true_mix);
        q_c_max = max(q_true_mix);
        
        % bisection method to calculate quantile of mixture
        f       = @(z) mixQ(q,z,kappa,mu_c_m,sigma_c_m);
        q_marg  = bisection(f,q_c_min-epsilon,q_c_max+epsilon,0,1e-10);     
    end
    if condq == 1
       q_cond      = zeros(n_fix,1);
       for j = 1:n_fix 
            q_cond(j) = quantiletrue(DGP,q,1,Sigma_M(idx_qfix:K,idx_qfix:K,:),mu_M(idx_qfix:K,:),Y_temp1(j),K-1,kappa,1);
       end
   end
    
% ---------------------------------------------------------------------

% ----- DGP6    
elseif DGP == 6
    % --- multivariate t-distribution
    for t = 1:T
        if gendata == 1
            u  = MVT_RND(mu, Sigma_U, df, 1)';
            % generate data
            y_MC(t,:) = u;
        end

        % true quantile
        if gentrueq == 1
            for idx = 1:K
                q_true(t,idx) = quantiletrue(DGP,q,idx,Sigma_U,mu,y_MC(t,:),K,0,0);
            end  
        end
    end
    if gentrueqfix == 1
        for t = 1:T_fix
           for j = 1:length(n_fix) 
               Y_temp      = [Y_temp1(t,j) n_fix(j)]; % set fixed value for other outcome variables   
               q_true_fix(t,j) = quantiletrue(DGP,q,idx_qfix,Sigma_U,mu,Y_temp,K,0,1);
           end
        end        
    end
    
    if marginalq == 1
       mu_c    = mu(idx_mar);
       sigma_c = Sigma_U(idx_mar,idx_mar);
       pd      = makedist('tLocationScale','mu',mu_c,'sigma',sqrt(sigma_c),'nu',df);
       q_marg  = icdf(pd,q); 
    end
    if condq == 1
       q_cond      = zeros(n_fix,1);
       for j = 1:n_fix 
            q_cond(j) = quantiletrue(DGP,q,1,Sigma_U(idx_qfix:K,idx_qfix:K),mu(idx_qfix:K),Y_temp1(j),K-1,0,1);
       end
    end
% ---------------------------------------------------------------------
% ----- DGP7    
elseif DGP == 7
    % --- conditional heteroskedasticity   
    for t = 1:T
        % generate data
        if gendata == 1
           z                = normrnd(0,1);
           u                = mvnrnd(mu,exp(z)*Sigma_U,1);
           y_MC(t,:)        = u;           
        end

        % true quantile
        if gentrueq == 1
            for idx = 1:K
                q_true(t,idx) = quantiletrue(DGP,q,idx,Omega,mu,y_MC(t,:),K,kappa,0);
            end
        end
    end
    if gentrueqfix == 1
        for t = 1:T_fix
           for j = 1:length(n_fix) 
               Y_temp  = [Y_temp1(t,j) n_fix(j)];
               q_true_fix(t,j) = quantiletrue(DGP,q,idx_qfix,Omega, mu,Y_temp,K,kappa,1);
           end 
        end    
    end
    if marginalq == 1
       for m = 1:size(Omega,3)     
           mu_c_m(m)     =  mu(idx_mar);
           sigma_c_m(m)  =  Omega(idx_mar,idx_mar,m);
           q_true_mix(m) =  norminv(q,mu_c_m(m),sqrt(sigma_c_m(m)));
       end
        q_c_min = min(q_true_mix);
        q_c_max = max(q_true_mix);
        % bisection method to calculate quantile of mixture
        f       = @(z) mixQ(q,z,kappa,mu_c_m,sigma_c_m);
        q_marg  = bisection(f,q_c_min-epsilon,q_c_max+epsilon,0,1e-10);     
    end
    if condq == 1
       q_cond      = zeros(n_fix,1);
       for j = 1:n_fix 
            q_cond(j) = quantiletrue(DGP,q,1,Omega(idx_qfix:K,idx_qfix:K,:),mu(idx_qfix:K),Y_temp1(j),K-1,kappa,1);
       end
    end

end    
    
if gendata == 1
    Y = y_MC;
else
    Y = 0;
end


