%--------------------------------------------------------------------------
% Code for On superlevel sets of conditional densities and multivariate quantile regression
% Simulation
% May 2024
% Annika Camehl, Dennis Fok, and Kathrin Gruber
%--------------------------------------------------------------------------
clear all; close all;
%% -------------------------| set DGP |------------------------------------
% DGP=1: u~Normal; 
% DGP=2: u~lognorm; 
% DGP=3: u~3-component mixture; 
% DGP=6: u~multi-variate t-distribution;
% DGP=7: conditional heteroskedasticity;

DGP      = 2;                  % DGP
A        = 'simulation';

%% -------------------------| Input |--------------------------------------
q_grid          = [0.2 0.4 0.6 0.8];  % quantiles
Tsim            = 10000;              % number of observations
T_fix           = 100;                % number of generated values for fixed quantiles   
T_q             = 99;                 % number of obersvations -1 for which quantile is calculated 
K               = 3;                  % number of variables (number of equations)
G               = 3;                  % number of explanatory variables
M               = 5;                  % maximal number of mixture components
nsave           = 50000;              % Number of MCMC draws to save
nburn           = 10000;              % Number of MCMC draws to discard
ntot            = nsave + nburn;      % Number of total MCMC draws
nMC             = 1000;               % Number of simulation draws
thin            = 10;                 % thinning
savemat         = 1;                  % save mat file if =1
save_figures    = 1;                  % save figures if =1    
epsilon         = 0.00001;

k               = G;                  % number of coefficients per equation
k_tot           = K*G;                % number of total coefficients


if savemat == 1
    results = ['DGP' num2str(DGP) '.mat']; 
end

%% -------------------------| Preparation |-------------------------------
% setting seed
rng(1);

T      = Tsim;
n_q    = length(q_grid);
n_fix  = size(q_grid,2);
Y_temp = zeros(T_fix,n_fix);


%% ---------------------| ouput matrices |---------------------------------

sigma_m_MC     = cell(nMC,1);
mu_m_MC        = cell(nMC,1);
kappa_MC       = cell(nMC,1);
T_m_MC         = cell(nMC,1);

Y_data_MC      = zeros(nMC,T,K);
q_true_MC      = zeros(nMC,T_q+1,K,n_q);
q_mix_MC       = zeros(nMC,T_q+1,K,n_q);
q_fix_MC       = zeros(nMC,T_fix,n_fix,n_q);
q_qr_c_MC      = zeros(nMC,T_q+1,K,n_q);
q_qr_cfix_MC   = zeros(nMC,T_fix,n_fix,n_q);
q_qr_cnl_MC    = zeros(nMC,T_q+1,K,n_q);
q_qr_cnlfix_MC = zeros(nMC,T_fix,n_fix,n_q);
sigma_qr_MC    = zeros(nMC,K,K,n_q);
u_qr_MC        = zeros(nMC,T,K,n_q);
y_fit_MC       = zeros(nMC,T_q+1,K,n_q);
y_fit_qr_MC    = zeros(nMC,T_q+1,K,n_q);
SE_MQreg       = zeros(nMC,T_q+1,K,n_q);
SE_qr          = zeros(nMC,T_q+1,K,n_q);
SE_qrc         = zeros(nMC,T_q+1,K,n_q);
SE_qrcnl       = zeros(nMC,T_q+1,K,n_q);
MSE_MQreg_MC   = zeros(nMC,K,n_q);
MSE_qr_MC      = zeros(nMC,K,n_q);
MSE_qrc_MC     = zeros(nMC,K,n_q);
MSE_qrcnl_MC   = zeros(nMC,K,n_q);

tic;

%% -------------------------| True Quantiles |----------------------------
q_true_fix_all  = zeros(T_fix,n_fix,length(q_grid));
q_marginal      = zeros(length(q_grid),1);
q_conditional   = zeros(length(q_grid),length(q_grid));

disp('calculating true quantiles')
 % --- calculate marginal quantile for y3
for idx_q = 1:length(q_grid)
    q                 = q_grid(idx_q); 
    [~,~,~, q_mar3]   = simdgp(0,0,0,1,0,DGP,T,K,q,n_fix,Y_temp,0,0,3);
    q_marginal(idx_q) = q_mar3;
end

% --- calculate conditional quantile for y2|y3
for idx_q = 1:length(q_grid)
    q                      = q_grid(idx_q); 
    [~,~,~,~, q_c2]        = simdgp(0,0,0,0,1,DGP,T,K,q,length(q_grid),q_marginal,0,2,0);
    q_conditional(:,idx_q) = q_c2;
end

% --- determine n_fix and Y_temp
n_fix  = q_marginal; % based on values for y3

for idx_f = 1:length(n_fix)
    Y_temp(:,idx_f) = linspace(q_conditional(idx_f,1),q_conditional(idx_f,length(n_fix)),T_fix); 
end
% --- calculate true quantiles with fixed values for conditional variables
idx_qfix = 1; % calculate true conditional quantile for first variable
for idx_q = 1:length(q_grid)
    q                         = q_grid(idx_q); 
    [~, ~,  q_true_fix]       = simdgp(0,0,1,0,0,DGP,T,K,q,n_fix,Y_temp,0,idx_qfix,0);
    q_true_fix_all(:,:,idx_q) = q_true_fix;
end

% ---- Simulation
for ndraws = 1:nMC
    ndraws
    T = Tsim;
    %% -------------------| simulate data |-------------------------------

    Y_data = simdgp(1,0,0,0,0,DGP,T,K,0,0,Y_temp,zeros(T,K),0,0);
      
    % --- data preparation for regression model
    y = Y_data;
    T = size(y,1);
    y_comp = y(:);
    
    q_true_all      = zeros(T_q+1,K,length(q_grid));
    % --- calculate true quantiles
    for idx_q = 1:length(q_grid)
        q                     = q_grid(idx_q); 
        [~, q_true]           = simdgp(0,1,0,0,0,DGP,T_q+1,K,q,n_fix,Y_temp,y((T-T_q):T,:),0,0);
        q_true_all(:,:,idx_q) = q_true;
    end
    
    %% --- OLS

    x_const   = ones(T,1); 
    Phi_OLS   = x_const\y;
    u_OLS     = y-x_const*Phi_OLS;
    sigma_OLS = u_OLS'*u_OLS./(T-(k+1));            
    sse_OLS   = u_OLS'*u_OLS;    
    
    %% --- univariate quantile regression
    % Regression Quantiles of Koenker and Bassett (1978), Econometrica

    run univariatequantile.m

    %% ---------------| Set priors |------------------------------------------

    run sc_prior.m

    %% ---------------| Algorithm |-------------------------------------------

    invSigma_m     = zeros(K,K,M);
    u_m            = cell(M,1);   
    Sigma_m        = zeros(K,K,M);
    sigma_m_draws  = zeros(nsave,K,K,M);
    mu_m_draws     = zeros(nsave,K,M);
    kappa_draws    = zeros(nsave,M);     
    T_m_draws      = zeros(nsave,M);
    w_m            = zeros(M,T,K,n_q);
    w              = zeros(M,T,K);
    accept         = zeros(ntot,M);
    
    T_m_new        = zeros(1,M);
    H_new          = zeros(T,M);
    rho_tilde_new  = zeros(1,M);
    kappa_new      = zeros(1,M);
    mu_m_new       = zeros(K,M);
    Sigma_m_new    = zeros(K,K,M);

   
    run sc_sampler.m
    
    %% ------------| Past processing of MCMC draws: |----------------------
    run sc_postprocessing.m
   
    %% ------------| quantile calculated with posterior means: |-----------
    disp('calculate quantile based on posterior means')
    Sigma_m_post   = squeeze(mean(sigma_m_draws_final,1));
    mu_m_post      = squeeze(mean(mu_m_draws_final,1));
    if size(mu_m_post,1) == 1
        mu_m_post = mu_m_post';
    end
    kappa_post     = squeeze(mean(kappa_draws_final,1))';

    [q_mix,q_fix] = calculationquantilesim(q_grid,K,Mfix,T,T_q,T_fix,n_fix,n_q,y,Y_temp,mu_m_post,Sigma_m_post,kappa_post); 
    
    % --- output
    sigma_m_MC{ndraws}            = squeeze(mean(sigma_m_draws_final,1));
    mu_m_MC{ndraws}               = squeeze(mean(mu_m_draws_final,1)); 
    kappa_MC{ndraws}              = mean(kappa_draws_final,1);
    T_m_MC{ndraws}                = mean(T_m_draws_final,1);
    q_mix_MC(ndraws,:,:,:)        = q_mix;
    q_fix_MC(ndraws,:,:,:)        = q_fix;
    q_true_MC(ndraws,:,:,:)       = q_true_all;
    % data
    Y_data_MC(ndraws,:,:)         = Y_data; 
    % models of comparison
    q_qr_c_MC(ndraws,:,:,:)       = q_qr_c_all;
    q_qr_cfix_MC(ndraws,:,:,:)    = q_qr_cfix_all;
    q_qr_cnl_MC(ndraws,:,:,:)     = q_qr_cnl_all;
    q_qr_cnlfix_MC(ndraws,:,:,:)  = q_qr_cnlfix_all;
    
    %% ----- Evaluation
    for idx_q = 1:n_q
        % --- fitted values
        y_fit_MQreg_temp = q_mix(:,:,idx_q);
        y_fit_qr_temp    = x_const((T-T_q):T,:)*Phi_qr(:,:,idx_q);
        % true values
        Y_q = q_true_all(:,:,idx_q);
        % we want to compare to the quantile
        Y_true = Y_q;

        % --- save fitted values
        y_fit_MC(ndraws,:,:,idx_q)     = y_fit_MQreg_temp;
        y_fit_qr_MC(ndraws,:,:,idx_q)  = y_fit_qr_temp;

        % --- squared errors (fitted values minus true values)
        SE_MQreg(ndraws,:,:,idx_q) = (y_fit_MQreg_temp - Y_true).*(y_fit_MQreg_temp - Y_true);
        SE_qr(ndraws,:,:,idx_q)    = (y_fit_qr_temp - Y_true).*(y_fit_qr_temp - Y_true); 
        SE_qrc(ndraws,:,:,idx_q)   = (q_qr_c_all(:,:,idx_q) - Y_true).*(q_qr_c_all(:,:,idx_q) - Y_true); 
        SE_qrcnl(ndraws,:,:,idx_q) = (q_qr_cnl_all(:,:,idx_q) - Y_true).*(q_qr_cnl_all(:,:,idx_q) - Y_true); 

        % --- MSE 
        MSE_MQreg_MC(ndraws,:,idx_q) = squeeze(mean(SE_MQreg(ndraws,:,:,idx_q),2));
        MSE_qr_MC(ndraws,:,idx_q)    = squeeze(mean(SE_qr(ndraws,:,:,idx_q),2));
        MSE_qrc_MC(ndraws,:,idx_q)   = squeeze(mean(SE_qrc(ndraws,:,:,idx_q),2));
        MSE_qrcnl_MC(ndraws,:,idx_q) = squeeze(mean(SE_qrcnl(ndraws,:,:,idx_q),2));
    end
    
end

runtime = toc;

% mean over all simulation draws
Y_fit_MQreg     = squeeze(mean(y_fit_MC,1));
Y_fit_qr        = squeeze(mean(y_fit_qr_MC,1));
Y_data_sim      = squeeze(mean(Y_data_MC,1));
q_true_mean     = squeeze(mean(q_true_MC,1));

% ----- average MSE (over simulation draws)
MSE_MQreg = mean(MSE_MQreg_MC,1);
MSE_qr    = mean(MSE_qr_MC,1);
MSE_qrc   = mean(MSE_qrc_MC,1);
MSE_qrcnl = mean(MSE_qrcnl_MC,1);

% ----- save

if savemat == 1
   save(results) 
end

%% --- Figure 3, 4, & Supplementary Appendix Figure 1, 2, and 3 
% depending on which DGP chosen
figure 
set(gcf, 'units','normalized','outerposition',[0 0 0.7 0.8]);
n_row = 3;
idx_select = 1:5;
ax = gobjects(12,1);
colorVec = {[0.0000,    0.0000,    0.0000],[0.2500,    0.2500,    0.2500],[0.5000,    0.5000,    0.5000], [0.7500 ,   0.7500,   0.7500]};
for col = 1:length(n_fix) 
    for row = 1:n_row
    i = (row-1)*length(n_fix)+col;
    ax(i) = subplot(n_row,length(n_fix),i); % last value gives position in plot, counts per row 
    idx_j = idx_select(col);
        for i_q = 1 : n_q
            % true
            temp1 = q_true_fix_all(:,idx_j,i_q)';
            temp2 = Y_temp(:,col)';
            temp_plot1  = [temp2' temp1'];
            plot(temp_plot1(:,1),temp_plot1(:,2),'Color', colorVec{i_q},  'LineWidth', 1.5) 
            hold on
            if row == 3
                % estimated with MQreg
                temp  = mean(q_fix_MC(:,:,idx_j,i_q),1);
                temp_plot2  = [temp2' temp'];
                plot(temp_plot2(:,1),temp_plot2(:,2),'Color', colorVec{i_q},  'LineStyle', '--', 'LineWidth', 1.5) 
                hold on
            elseif row == 1
                % estimated with cQreg
                temp3  = mean(q_qr_cfix_MC(:,:,idx_j,i_q),1);
                temp_plot3  = [temp2' temp3'];
                plot(temp_plot3(:,1),temp_plot3(:,2),'Color', colorVec{i_q},  'LineStyle', '-.', 'LineWidth', 1.5) 
                hold on
            elseif row == 2
                % estimated with cQreg with squared terms
                temp4  = mean(q_qr_cnlfix_MC(:,:,idx_j,i_q),1);
                temp_plot4  = [temp2' temp4'];
                plot(temp_plot4(:,1),temp_plot4(:,2),'Color', colorVec{i_q},  'LineStyle', '-.', 'LineWidth', 1.5) 
                hold on
            end
        end 
        if row == 3 && col == 1
            ylabel({'\fontsize{13}MQReg';'\fontsize{10}Q(y_1|y_2,y_3)'})
        elseif row == 1 && col == 1
            ylabel({'\fontsize{13}QReg with y_{(-k)}';'\fontsize{10}Q(y_1|y_2,y_3)'})
        elseif row == 2 && col == 1
            ylabel({'\fontsize{13}QReg with y_{(-k)}, (y_{(-k)})^2';'\fontsize{10}Q(y_1|y_2,y_3)'})
        else
           ylabel('Q(y_1|y_2,y_3)', 'FontSize', 10) 
        end
        title(['y_3 = ' num2str(round(n_fix(idx_j),2))], 'FontSize', 12)
        xlabel('y_2', 'FontSize', 10) 
        axis tight
        grid on
        hold on
    end
end
linkaxes(ax, 'y')
if save_figures == 1
    h1=gcf;
    file_name = ['condQ' 'DGP' num2str(DGP)];
    print(h1,'-depsc',file_name)
end

%% --- Table 2: one column (for one DGP)
% Mean squared errors for simulated multivariate response distributions
MSE_all = zeros(n_q,4);
MSE_all(:,1) = round(mean(squeeze(MSE_qr)),3)';
MSE_all(:,2) = round(mean(squeeze(MSE_qrc)),3)';
MSE_all(:,3) = round(mean(squeeze(MSE_qrcnl)),3)';
MSE_all(:,4) = round(mean(squeeze(MSE_MQreg)),3)';
MSE_all

