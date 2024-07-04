    

%% --- univariate quantile regression
% Regression Quantiles of Koenker and Bassett (1978), Econometrica
X_temp   = [0 0 0];
x        = zeros(T,G);
Phi_qr   = zeros(1,K,n_q);
sigma_qr = zeros(K,K,n_q);
u_qr     = zeros(T,K,n_q);
q_qr_c_all       = zeros(T_q+1,K,n_q);
q_qr_cfix_all    = zeros(T_fix,length(n_fix),n_q);
q_qr_cnl_all     = zeros(T_q+1,K,n_q);
q_qr_cnlfix_all  = zeros(T_fix,length(n_fix),n_q);

% --- conditional quantile based on univariate quantile regression
% calculate conditional quantiles for qr by includung y_{-k} in x

for idx_q = 1:length(q_grid)
    q                    = q_grid(idx_q); 
    
    for i = 1:K
        Phi_qr(:,i,idx_q) = rq_fnm(x_const((T-T_q):T,:), y((T-T_q):T,i),q);
    end
    %Phi_qr(:,:,idx_q)    = quantilereguv(y,x_const,q,K,G,1);
    u_qr(:,:,idx_q)      = y-x_const*squeeze(Phi_qr(:,:,idx_q));
    sigma_qr(:,:,idx_q)  = squeeze(u_qr(:,:,idx_q))'*squeeze(u_qr(:,:,idx_q))./(T-k); 
    for i_var = 1:K
        idx_qr                    = 1:K;
        idx_qr(i_var)             = []; 
        y_qr                      = y(:,idx_qr);
        % with y_{-k} in x
        x_qr                      = [x, y_qr, ones(T,1)];
        Phi_C_qr                  = rq_fnm(x_qr, y(:,i_var),q);
        q_qr_c_all(:,i_var,idx_q) = x_qr((T-T_q):T,:)*Phi_C_qr;
        % with y_{-k} and y^2 in x
        x_qrnl                      = [x, y_qr, y_qr.^2, ones(T,1)]; %[x, y_qr, y.^2, ones(T,1)];
        Phi_Cnl_qr                  = rq_fnm(x_qrnl, y(:,i_var),q);
        q_qr_cnl_all(:,i_var,idx_q) = x_qrnl((T-T_q):T,:)*Phi_Cnl_qr;
        if i_var == 1
            for j = 1:length(n_fix)
                q_qr_cfix_all(:,j,idx_q)   = [X_temp.*ones(T_fix,G) Y_temp(:,j) n_fix(j)*ones(T_fix,1) ones(T_fix,1)]*Phi_C_qr;
                q_qr_cnlfix_all(:,j,idx_q) = [X_temp.*ones(T_fix,G) Y_temp(:,j) n_fix(j)*ones(T_fix,1) Y_temp(:,j).^2 (n_fix(j)*ones(T_fix,1)).^2 ones(T_fix,1)]*Phi_Cnl_qr;
            end
        end
    end

end