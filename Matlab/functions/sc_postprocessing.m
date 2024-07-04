% ----- Past processing of MCMC draws:
% --- Step A: determine number of non-empty components
nsavethin = nsave/thin; % nsave draws - thinning

M0 = zeros(nsavethin,1);
for irep = 1:nsavethin
    temp = T_m_draws(irep,:);
    M0(irep) = M - sum(temp(:)<=0); % number of non-empty components
end

% --- Step B: estimate true number of mixture components
Mfix = mode(M0); % estimated number of mixture components

% --- Step C: remove draws
% C.1 remove MCMC iterations where the number of non-empty componets is not equal to Mfix
% C.2 remove draws from empty components
sigma_m_draws_temp   = [];
mu_m_draws_temp      = [];
kappa_draws_temp     = [];
T_m_draws_temp       = [];
sigma_m_draws_final  = [];
mu_m_draws_final     = [];
kappa_draws_final    = [];
T_m_draws_final      = [];
idx_cs_final         = [];
idx_per   = 0;
 for irep = 1:nsavethin
    % remove draws where the number of non-empty components is not equal to Mfix
    if M0(irep) == Mfix 
        idx_per = idx_per+1;
        idx_m = 0;
        for m = 1:M
            % remove empty components 
            if T_m_draws(irep,m)>0
                idx_m = idx_m+1;
                sigma_m_draws_temp(idx_per,:,:,idx_m) = sigma_m_draws(irep,:,:,m);
                mu_m_draws_temp(idx_per,:,idx_m)      = mu_m_draws(irep,:,m);
                kappa_draws_temp(idx_per,idx_m)       = kappa_draws(irep,m); 
                T_m_draws_temp(idx_per,idx_m)         = T_m_draws(irep,m);
            end
        end
    end
 end

if strcmp(A, 'application') == 1
    % --- Step D: clustering based on component means 
    rep = 0;
    idx_draws_mu  = 0;

    while idx_draws_mu <= round(0.4*(nsave/thin)) % try a few times until a sufficient number is kept
        rep = rep +1;
        % based on mean
        mu_temp = zeros(Mfix*idx_per,K);
        for idx_var = 1:K
            mu_temp(:,idx_var) = reshape(mu_m_draws_temp(:,idx_var,:),Mfix*idx_per,1);
        end
        idx_c_mu = kmedoids(mu_temp,Mfix,'Distance','mahalanobis'); % K-centroids clustering (based on Mahalanobis distance)
        
        % check for which more draws are kept
        idx_cs_mu     = reshape(idx_c_mu,idx_per,Mfix); % reshape clustering indicators
        idx_draws_mu  = 0;
        
        for irep = 1:idx_per
            tempmu  = idx_cs_mu(irep,:);
            % keep draws where classification is a permuation of 1:Mfix
            if numel(unique(tempmu)) == Mfix 
                idx_draws_mu = idx_draws_mu+1;      
            end
        end
        if rep == 20
            disp('no clustering based on mean possible after 20 repetitions')
            break
            
        end
    end
    disp([num2str(rep) ' trys for clustering based on mean'])
    idx_c = idx_c_mu;
else
    % --- Step D: clustering based on component means or on variance
    % based on mean
    mu_temp = zeros(Mfix*idx_per,K);
    for idx_var = 1:K
        mu_temp(:,idx_var) = reshape(mu_m_draws_temp(:,idx_var,:),Mfix*idx_per,1);
    end
    idx_c_mu = kmedoids(mu_temp,Mfix,'Distance','mahalanobis'); % K-centroids clustering (based on Mahalanobis distance)
    
    % based on variance
    sigma_temp = zeros(Mfix*idx_per,K);
    for idx_var = 1:K
        sigma_temp(:,idx_var) = reshape(log(sigma_m_draws_temp(:,idx_var,idx_var,:)),Mfix*idx_per,1);
    end
    idx_c_var = kmedoids(sigma_temp,Mfix,'Distance','seuclidean'); % K-centroids clustering (based on Mahalanobis distance)

    % check for which more draws are kept
    idx_cs_mu     = reshape(idx_c_mu,idx_per,Mfix); % reshape clustering indicators
    idx_cs_var    = reshape(idx_c_var,idx_per,Mfix); % reshape clustering indicators
    idx_draws_mu  = 0;
    idx_draws_var = 0;
    
    for irep = 1:idx_per
        tempmu  = idx_cs_mu(irep,:);
        tempvar = idx_cs_var(irep,:);
        % keep draws where classification is a permuation of 1:Mfix
        if numel(unique(tempmu)) == Mfix 
            idx_draws_mu = idx_draws_mu+1;      
        end
        if numel(unique(tempvar)) == Mfix 
            idx_draws_var = idx_draws_var+1;      
        end
    end
    
    % use the clustering method which keeps the most draws
    if idx_draws_var <= 0.5*idx_per
       idx_c = idx_c_mu; 
       disp('clustering based on mean')
    else
       idx_c = idx_c_var; 
       disp('clustering based on variance')
    end
end

% --- Step E: unique labeling
% construct classification sequence
idx_cs = reshape(idx_c,idx_per,Mfix); % reshape clustering indicators
idx_draws = 0;
for irep = 1:idx_per
    temp = idx_cs(irep,:);
    % keep draws where classification is a permuation of 1:Mfix
    if numel(unique(temp)) == Mfix 
        idx_draws = idx_draws+1;
        idx_cs_final(idx_draws,:) = idx_cs(irep,:);
        % resorting draws according to the classification sequences
        for mfix = 1:Mfix
            idx_mfix = find(idx_cs_final(idx_draws,:) == mfix);
            sigma_m_draws_final(idx_draws,:,:,mfix) = sigma_m_draws_temp(irep,:,:,idx_mfix);
            mu_m_draws_final(idx_draws,:,mfix)      = mu_m_draws_temp(irep,:,idx_mfix);
            kappa_draws_final(idx_draws,mfix)       = kappa_draws_temp(irep,idx_mfix); 
            T_m_draws_final(idx_draws,mfix)         = T_m_draws_temp(irep,idx_mfix);
        end
    end
end