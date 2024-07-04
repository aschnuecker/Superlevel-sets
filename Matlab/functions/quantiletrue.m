function [q_true] = quantiletrue(DGP,q,idx,Sigma_U,mu_M,Y,K,kappa,fix)
epsilon = 0.0001;

if DGP == 1
    idx_ex      = 1:K;
    idx_ex(idx) = [];
    temp_cov    = Sigma_U(idx,idx_ex);
    temp_siginv = inv(Sigma_U(idx_ex,idx_ex));

    % conditional moments
    mu = mu_M;
    if fix == 1
       mu_c = mu(idx) + (Y - mu(idx_ex)')*temp_siginv*temp_cov';
    else
       mu_c = mu(idx) + (Y(idx_ex) - mu(idx_ex)')*temp_siginv*temp_cov';
    end
    sigma_c = Sigma_U(idx,idx) - temp_cov*temp_siginv*temp_cov';
    q_true  = norminv(q,mu_c,sqrt(sigma_c));    
    
elseif DGP == 2
    idx_ex      = 1:K;
    idx_ex(idx) = [];
    temp_cov    = Sigma_U(idx,idx_ex);
    temp_siginv = inv(Sigma_U(idx_ex,idx_ex));
    
    mu = mu_M ;
    
    if fix == 0
      Y_temp = Y(idx_ex);
    else
      Y_temp = Y;
    end

    if any(Y_temp<=0)
        q_true  = zeros(K,idx);
    else  
        mu_c    = mu(idx) + (log(Y_temp) - mu(idx_ex)')*temp_siginv*temp_cov';       
        sigma_c = Sigma_U(idx,idx) - temp_cov*temp_siginv*temp_cov';
        q_true  = exp(norminv(q,mu_c,sqrt(sigma_c)));
    end

elseif DGP == 3
    for m = 1:3
       mu          = mu_M(:,m);
       idx_ex      = 1:K;
       idx_ex(idx) = [];
       temp_cov    = Sigma_U(idx,idx_ex,m);
       temp_siginv = inv(Sigma_U(idx_ex,idx_ex,m));
       if fix == 0
          Y_temp = Y(idx_ex);
       else
          Y_temp = Y;
       end
       % conditional moments
       mu_c_m(m)     =  mu(idx) + (Y_temp - mu(idx_ex)')*temp_siginv*temp_cov';
       sigma_c_m(m)  =  Sigma_U(idx,idx,m) - temp_cov*temp_siginv*temp_cov';
       q_true_mix(m) =  norminv(q,mu_c_m(m),sqrt(sigma_c_m(m)));
       % conditional weight
       w_m(m) = kappa(m)*mvnpdf(Y_temp,mu(idx_ex)',Sigma_U(idx_ex,idx_ex,m));
    end
    q_c_min = min(q_true_mix);
    q_c_max = max(q_true_mix);
    w       = w_m./sum(w_m);  
    % bisection method to calculate quantile of mixture
    f       = @(z) mixQ(q,z,(w)',(mu_c_m)',sigma_c_m);
    q_true  = bisection(f,q_c_min-epsilon,q_c_max+epsilon,0,1e-10);



 elseif DGP == 6 % multivariate t
    df          = 5;
    mu          = mu_M;
    idx_ex      = 1:K;
    idx_ex(idx) = [];
    temp_cov    = Sigma_U(idx,idx_ex);
    temp_siginv = inv(Sigma_U(idx_ex,idx_ex));

    % conditional moments
    if fix == 0
      Y_temp = Y(idx_ex);
    else
      Y_temp = Y;
    end
    mu_c = mu(idx) + (Y_temp - mu(idx_ex)')*temp_siginv*temp_cov';
    sigma_c = (df+(Y_temp - mu(idx_ex)')*temp_siginv*(Y_temp - mu(idx_ex)')')/(df+(K-1))*(Sigma_U(idx,idx) - temp_cov*temp_siginv*temp_cov');
    df_c    = df+(K-1);
    pd      = makedist('tLocationScale','mu',mu_c,'sigma',sqrt(sigma_c),'nu',df_c);
    q_true  = icdf(pd,q);    
    
elseif DGP == 7 
    for m = 1:size(Sigma_U,3)        
       mu          = mu_M;
       idx_ex      = 1:K;
       idx_ex(idx) = [];
       temp_cov    = Sigma_U(idx,idx_ex,m);
       temp_siginv = inv(Sigma_U(idx_ex,idx_ex,m));
       if fix == 0
          Y_temp = Y(idx_ex);
       else
          Y_temp = Y;
       end
       % conditional moments
       mu_c_m(m)     =  mu(idx) + (Y_temp - mu(idx_ex)')*temp_siginv*temp_cov';
       sigma_c_m(m)  =  Sigma_U(idx,idx,m) - temp_cov*temp_siginv*temp_cov';
       q_true_mix(m) =  norminv(q,mu_c_m(m),sqrt(sigma_c_m(m)));
       % conditional weight
       w_m(m) = kappa(m)*mvnpdf(Y_temp,mu(idx_ex)',Sigma_U(idx_ex,idx_ex,m));
    end
    q_c_min = min(q_true_mix);
    q_c_max = max(q_true_mix);
    w       = w_m./sum(w_m);  
    % bisection method to calculate quantile of mixture
    f       = @(z) mixQ(q,z,(w)',(mu_c_m)',sigma_c_m);
    q_true  = bisection(f,q_c_min-epsilon,q_c_max+epsilon,0,1e-10);

end
    