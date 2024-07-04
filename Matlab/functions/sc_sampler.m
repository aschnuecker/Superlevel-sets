%% posterior sampler

% Starting values
% Start with coefficients drawn from posterior with uninformative prior
for m = 1:M
    y_m = y(H(:,m)==1,:);        % select observations
    x_m = x(H(:,m)==1,:); 

    y_m_comp = y_m(:);
    x_m_comp = kron(eye(K),x_m);

    %% --- OLS
    if T_m(m) < 2
       Sigma      = inv(wishrnd(inv(S0),s0+K+1));
       cinv_sigma = chol(Sigma)\eye(K);
       invSigma   = inv(Sigma);

    else

        x_m_const = ones(T_m(m),1);
        Phi_OLS_m = inv(x_m_const'*x_m_const)*x_m_const'*y_m;
        u_OLS_m   = y_m-x_m_const*Phi_OLS_m;
        sse_m     = u_OLS_m'*u_OLS_m;  
        Sigma     = inv(wishrnd(inv(sse_m),T_m(m)+K+2));  
        cinv_sigma = chol(Sigma)\eye(K);
        invSigma   = inv(Sigma);
    end

    Sigma_m(:,:,m)    = Sigma;
    invSigma_m(:,:,m) = invSigma;
end

%% Algorithm

disp('estimating mixture model')
for irep = 1:ntot
        if mod(irep,1000)==0
            tocmin = toc/60 ;
            fprintf('%d minutes for %d draws of %d total draws.\n',tocmin,irep,ntot);
         end

    % --- Step 1: conditional on H:
    % --- Step 1a: sample weights kappa from Dirichlet distribution         
    atemp = gamrnd(rho_tilde,M)';
    kappa = atemp./sum(atemp);

    % ----------------| Draw reduced form |----------------------------
    % --- Sample conditional on the classification
   for m = 1:M
        %% --- if number of observations for component is >0
        if T_m(m) > 0
            y_m = y(H(:,m)==1,:);        % select observations 
            y_m_comp = y_m(:);
            u = y_m;
            muall  = repmat(mu_m(:,m)',T_m(m),1);
            mut    = muall(:);
            % --- Step 1b: draw Sigma_m
            S1 = S0 + (u-muall)'*(u-muall);
            s1 = s0 + T_m(m); 
            Sigma_m(:,:,m)    = inv(wishrnd(inv(S1), s1+K+1));
            invSigma_m(:,:,m) = squeeze(inv(Sigma_m(:,:,m)));
            u_m{m}(:,:) = y_m;

            % --- Step 1d: draw mu_m             
            mu_B      = inv(inv(mu_B0)+T_m(m)*invSigma_m(:,:,m));
            mu_b      = mu_B*(mu_B0\mu_b0 + invSigma_m(:,:,m)*T_m(m)*mean(u,1)');  
            mu_m(:,m) = mu_b + chol(mu_B)'*randn(K,1);             

            % --- set pi
            % mu of y_t
            temp_mu = mu_m(:,m)';
            pi(m,:) = kappa(m)*mvnpdf(y,temp_mu,Sigma_m(:,:,m));

        else
            %% --- if component has <=0 observations: sample from prior
            muall     = repmat(mu_m(:,m)',T_m(m),1);
            mut       = muall(:);

            % --- Step 1b: draw Sigma_m
            Sigma_m(:,:,m) = inv(wishrnd(inv(S0),s0+K+1)); 

            % --- Step 1d: draw mu_m
            mu_m(:,m) = mu_b0 + chol(mu_B0)'*randn(K,1);

            % --- set pi
            % mu of y_t
            temp_mu = mu_m(:,m)';
            pi(m,:) = kappa(m)*mvnpdf(y,temp_mu,Sigma_m(:,:,m));            

        end        

   end
    
    % --- Step 2: Classify each observation
    % --- sample H from Multinomial        
    pi_tilde = pi./sum(pi);
    pi_tilde(isnan(pi_tilde))=0;
    H = mnrnd(1,pi_tilde');
    T_m = sum(H)'; % number of observation per component m

    % --- Step 3: Sample hyperparameter        
    % --- hyperparameter for mu
    for idx_var = 1:K
        b = sum(((mu_m(idx_var,:)-mu_b0(idx_var)).*(mu_m(idx_var,:)-mu_b0(idx_var)))/R0(idx_var,idx_var));
        lambda(idx_var) = gigrnd(v1-(M/2), 2*v2, b, 1);
    end
    Lambda = diag(sqrt(lambda));
    mu_B0  = Lambda*R0*Lambda;
    mu_b0  = mean(mu_m,2) + chol((1/M)*mu_B0)'*randn(K,1);

    % --- sample rho
    for m = 1:M

        % set proposal value for rho
        proposal = normrnd(0,sd_p);
        rho_p(m) = exp(log(rho(m))+proposal);

        % loglikelihood ratio of rho_p and rho
        % l_rho = loglik(m,rho_p,rho,kappa,M,beta_01,beta_02);
        if kappa(m) == 0
          l_rho = (rho_p(m)-rho(m))*log(1.0000e-50)+gammaln(rho(m))-gammaln(rho_p(m))...
                +gammaln(sum(rho_p))-gammaln(sum(rho))...
                +(beta_01-1)*(log(rho_p(m))-log(rho(m)))...
                -beta_02*M*(rho_p(m)-rho(m))+log(rho_p(m))-log(rho(m));
        else
          l_rho = (rho_p(m)-rho(m))*log(kappa(m))+gammaln(rho(m))-gammaln(rho_p(m))...
                +gammaln(sum(rho_p))-gammaln(sum(rho))...
                +(beta_01-1)*(log(rho_p(m))-log(rho(m)))...
                -beta_02*M*(rho_p(m)-rho(m))+log(rho_p(m))-log(rho(m));

        end

        a_1 = min(exp(l_rho),1);
        a_2 = rand(1);

        % accept or reject proposal
        if a_2 <= a_1
            rho(m)         = rho_p(m);
            accept(irep,m) = 1;
        else 
            accept(irep,m) = 0;
        end               
        rho_tilde(m) = rho(m) + T_m(m);

    end

    % --- Step 4: permutation step
    m_per = randperm(M); % random permutation
    for m = 1:M
        idx_m                = m_per(m);
        T_m_new(m)           = T_m(idx_m);
        H_new(:,m)           = H(:,idx_m);
        rho_tilde_new(m)     = rho_tilde(idx_m);
        kappa_new(m)         = kappa(idx_m);
        mu_m_new(:,m)        = mu_m(:,idx_m);
        Sigma_m_new(:,:,m)   = Sigma_m(:,:,idx_m);
    end
    T_m         = T_m_new;
    H           = H_new;
    rho_tilde   = rho_tilde_new;
    kappa       = kappa_new;
    mu_m        = mu_m_new;
    Sigma_m     = Sigma_m_new;

    % clear objects
    u_m        = cell(M,1); 
    
    % ---------------------| save draws |-------------------------------
    sigma_m_alldraws(irep,:,:,:) = Sigma_m;
    mu_m_alldraws(irep,:,:)      = mu_m;
    kappa_alldraws(irep,:)       = kappa; 
    T_m_alldraws(irep,:)         = T_m;
    % draws after burn in
    if irep > nburn
        sigma_m_draws(irep-nburn,:,:,:) = Sigma_m;
        mu_m_draws(irep-nburn,:,:)      = mu_m;
        kappa_draws(irep-nburn,:)       = kappa;
        T_m_draws(irep-nburn,:)         = T_m;
    end
end
% thinning
sigma_m_draws   = sigma_m_draws(1:thin:end,:,:,:);
mu_m_draws      = mu_m_draws(1:thin:end,:,:);
kappa_draws     = kappa_draws(1:thin:end,:);
T_m_draws       = T_m_draws(1:thin:end,:);

runtime = toc;