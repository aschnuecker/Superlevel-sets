%% prior distributions
% --- prior for K_t (classify observations)
pi = zeros(M,T);
pi_tilde = (1/M)*ones(M,1);
H = zeros(T,M); % indicator observations per component m
for t = 1:T
    H(t,:) = mnrnd(1,pi_tilde);
end
T_m = sum(H)'; % number of observation per component m

% --- prior for kappa: Dirichlet(rho_0.1,...,rho_0.M)
rho_tilde = zeros(M,1);
rho_p     = zeros(M,1);
atemp     = zeros(M,1);
rho       = zeros(M,1);
beta_01   = 10; 
beta_02   = 40; 
sd_p      = 0.5;  % standard deviation of proposal density

for m = 1:M
    rho(m)       = gamrnd(beta_01,1./(beta_02*M));
    rho_tilde(m) = rho(m);
end

for m = 1:M
    atemp(m) = gamrnd(rho_tilde(m),1);
end
kappa = atemp./sum(atemp);

% --- prior for Sigma: inverse Wishart
s0 = 1;        % prior degrees of freedom 
S0 = eye(K);   % (inverse) prior mean (inv. wishart)

% --- prior for mu
lambda = zeros(K,1);
v1 = 0.5; 
v2 = 0.5;
for idx_var = 1:K
    lambda(idx_var) = gamrnd(v1,1/v2);
end
Lambda = diag(sqrt(lambda));

if strcmp(A, 'application') == 1
    R0     = diag(range(y).*range(y));
else
    R0     = eye(K);
end

mu_B0  = Lambda*R0*Lambda;
mu_b0  = 0*ones(K,1);


mu_m       = zeros(K,M);
for m = 1:M
    mu_m(:,m) = mu_b0 + chol(mu_B0)'*randn(K,1);
end