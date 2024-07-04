%--------------------------------------------------------------------------
% Code for On superlevel sets of conditional densities and multivariate quantile regression
% Application
% May 2024
% Annika Camehl, Dennis Fok, and Kathrin Gruber
%--------------------------------------------------------------------------
clear all; close all;

%% -------------------------| Input |------------------------------------
A = 'application';
addpath('functions')

q_grid   = [0.2 0.4 0.6 0.8];   % quantiles
M        = 5;                   % maximal number of mixture components
nsave    = 200000;              % Number of MCMC draws to save
nburn    = 400000;              % Number of MCMC draws to discard
ntot     = nsave + nburn;       % Number of total MCMC draws
thin     = 40;                  % thinning

savemat         = 1;            % save mat file if =1
save_figures    = 1;            % save figures if =1    

quantileperdraw = 1;            % calculate quantiles per darw if =1, otherwise based on posterior estimates
postprocess     = 1;            % post-processing of MCMC draws if =1

if savemat == 1
    results = [A '.mat']; 
end

%% -------------------------| Load Data |---------------------------------
Dataraw      = readtable('CEX_2015_b.csv');
varnames_raw = Dataraw.Properties.VariableNames;
coluse       = [15 16 17 19];
varnames     = varnames_raw(coluse);

Data = table2array(Dataraw(:,coluse));
Yraw = Data;
mean_raw = mean(Yraw+1);
std_raw  = std(Yraw+1);

if sum(any(Data<0))>0 
   disp('data contains negative values') 
   % remove rows with negative values
   Yraw = Yraw(all(Yraw>0,2),:);
end

%% FIGURE 5: 
% Empirical distribution of the three expenditures categories food, housing, and
% utilities conditional on income before taxes
% --- scatter plot
varnames_plot = {'income' 'food' 'housing' 'utility'};

figure;
set(gcf, 'units','normalized','outerposition',[0 0 0.8 0.5]);

i = 0;
for j = 2: length(coluse)
   i = i + 1;
   subplot(1, length(coluse)-1,i)
   scatter(Yraw(:,1),Yraw(:,j),'filled');
   grid on
   set(gca,'FontSize',14)
   ylabel(varnames_plot(j))
   xlabel(varnames_plot(1))
   
end
if save_figures == 1
    h1 = gcf;
    set(h1,'Units','Inches');
    pos = get(h1,'Position');
    set(h1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    file_name = 'scatterdata';
    saveas(h1,file_name, 'epsc')
    clear 'h1'
end

%% Data transformation
factor      = 1;           % since zeros in data set add 1 to all series
delta       = (500*12);    % change in explanatory variable

% take log of data
Y_data      = log(Yraw+factor);
stdlogdata  = std(Y_data);
meanlogdata = mean(Y_data);
% normalization
Y_data      = normalize(Y_data);


T      = size(Y_data,1);
K      = size(Y_data,2);     % number of equations
k      = K;                  % number of coefficients per equation
k_tot  = K*K;                % number of total coefficients


%% -------------------------| Preparation |-------------------------------
n_q = length(q_grid);

rng(1);

tic;

% --- data preparation for regression model
y = Y_data;
x = zeros(T,K);

y_comp = y(:);
x_comp = kron(eye(K),x);

%% --- OLS
x_const   = [x, ones(T,1)]; 
Phi_OLS   = x_const\y;
u_OLS     = y-x_const*Phi_OLS;
sigma_OLS = u_OLS'*u_OLS./(T-(k+1));           
sse_OLS   = u_OLS'*u_OLS;    

%% ---------------------| ouput matrices |---------------------------------
Tfix                 = 100;

Sigma_m              = zeros(K,K,M);
invSigma_m           = zeros(K,K,M);
u_m                  = cell(M,1); 

sigma_m_draws        = zeros(nsave,K,K,M);
mu_m_draws           = zeros(nsave,K,M);
kappa_draws          = zeros(nsave,M);     
q_marginal_draws     = zeros(nsave/thin,Tfix);
q_mix_draws          = zeros(nsave/thin,Tfix,K-1,n_q);
q_mix_mar_draws      = zeros(nsave/thin,Tfix,K-1,n_q);

sigma_m_alldraws     = zeros(ntot,K,K,M);
mu_m_alldraws        = zeros(ntot,K,M);
kappa_alldraws       = zeros(ntot,M); 
T_m_alldraws         = zeros(ntot,M); 

T_m_draws            = zeros(nsave,M);
accept               = zeros(ntot,M);

T_m_new              = zeros(1,M);
H_new                = zeros(T,M);
rho_tilde_new        = zeros(1,M);
kappa_new            = zeros(1,M);
mu_m_new             = zeros(K,M);
Sigma_m_new          = zeros(K,K,M);

%% ---------------| Set priors |------------------------------------------

run sc_prior.m

%% ---------------| Algorithm |-------------------------------------------

run sc_sampler.m

%% ---------------| conditional quantiles per draw |----------------------
% calculate quantiles for draws after burn-in
if quantileperdraw == 1 
    disp('calculate quantile per draw') 
    rt = tic;
    parfor idx_draws = 1:size(sigma_m_draws,1)
        if mod(idx_draws,100)==0
            tocmin = toc(rt)/60 ;
            fprintf('%d minutes for %d draws of %d total draws.\n',tocmin,idx_draws,size(sigma_m_draws,1));
        end
        Sigma_m_drawtemp   = squeeze(sigma_m_draws(idx_draws,:,:,:));
        mu_m_drawtemp      = squeeze(mu_m_draws(idx_draws,:,:));
        if size(mu_m_drawtemp,1) == 1
            mu_m_drawtemp  = mu_m_drawtemp';
        end
        kappa_drawtemp     = squeeze(kappa_draws(idx_draws,:))';
        
        [q_marginal_draws(idx_draws,:),q_mix_draws(idx_draws,:,:,:),q_mix_mar_draws(idx_draws,:,:,:)] = calculationquantile(stdlogdata,meanlogdata,q_grid,K,M,Tfix,n_q,mu_m_drawtemp,Sigma_m_drawtemp,kappa_drawtemp,delta)
  
    end
end

%% ------------| Past processing of MCMC draws: |--------------------------
if postprocess == 1
    run sc_postprocessing.m
end

%% ------------| quantile calculated with posterior means: |---------------
disp('calculate quantile based on posterior means')
Sigma_m_post   = squeeze(mean(sigma_m_draws_final,1));
mu_m_post      = squeeze(mean(mu_m_draws_final,1));
if size(mu_m_post,1) == 1
    mu_m_post = mu_m_post';
end
kappa_post     = squeeze(mean(kappa_draws_final,1))';

[q_marginal,q_mix,q_mix_mar] = calculationquantile(stdlogdata,meanlogdata,q_grid,K,Mfix,Tfix,n_q,mu_m_post,Sigma_m_post,kappa_post,delta); 


%% ----- save
 if savemat == 1
    save(results)          % save output
    save('inputJulia.mat', "stdlogdata","meanlogdata","mu_m_post", "Sigma_m_post", "kappa_post") % save input for Julia code
 end


%% FIGURE 7: 
% Quantile-varying marginal effects for the three expenditures categories food,
% housing, and utilities conditional on different levels of income
q_grid_fix      = linspace(0.1,0.9,Tfix);

% --- plot: marginal effects based on quantiles per draw with bands 
if quantileperdraw == 1
    % marginal effect:
    cmap  =  {[0.0000,    0.0000,    0.0000],[0.2500,    0.2500,    0.2500],[0.5000,    0.5000,    0.5000], [0.7500 ,   0.7500,   0.7500]};
    cmap2 =  {[0.3000,    0.3000,    0.3000],[0.4500,    0.4500,    0.4500],[0.7000,    0.7000,    0.7000], [0.8500 ,   0.8500,   0.8500]};

    ptile = [0.05 0.16 0.50 0.84 0.95];
     
    figure;
    set(gcf, 'units','normalized','outerposition',[0 0 1.0 0.7]);

    i = 0;
    for j = 2:K
       i = i + 1;
       subplot(1,K-1,i)
       grid on
       beta_q_draws = exp(squeeze(q_mix_mar_draws(:,:,i,:))) - exp(squeeze(q_mix_draws(:,:,i,:)));        
       q_mixtemp    = squeeze(quantile(beta_q_draws,ptile));
       for ii = n_q:-1:1
           shadedplot(q_grid_fix,squeeze(q_mixtemp(1,:,ii)),squeeze(q_mixtemp(5,:,ii)),cmap2{ii},cmap2{ii});
           hold all
           hold on
       end

       for ii = 1 : n_q
           % median response
           h(ii) = plot(q_grid_fix,squeeze(q_mixtemp(3,:,ii)),'Linewidth',2,'Color',cmap{ii});
           hold on
       end
       
       
       set(gca,'FontSize',14)
       ylabel('average marginal effect')
       xlabel('marginal quantiles income')
       title(varnames_plot{j})
       legend([h(1) h(2) h(3) h(4)],[varnames_plot{j} ' ' 'cond q20'],[varnames_plot{j} ' '  'cond q40'], [varnames_plot{j} ' ' 'cond q60'], [varnames_plot{j} ' ' 'cond q80'],'Location','southoutside')
    end
    if save_figures == 1
        h1 = gcf;
        set(h1,'Units','Inches');
        pos = get(h1,'Position');
        set(h1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
        file_name = 'marginaleffects';
        saveas(h1,file_name, 'epsc')
    end
end


