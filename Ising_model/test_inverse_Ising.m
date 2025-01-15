% 用预定义的模拟作为拟合目标检验拟合效果
clear

%% 基于FC的预定义模型
T = 1.5;
J = readmatrix("..\demo\input\Desikan_input\fc_train.csv");
N = length(J);
J = -inv(J)./2;
J(eye(N)==1) = 0;

h = 0.1*ones([N,1])+0.1*(rand([N,1])-0.5);
steps = 10000;

target_series = simulate_Ising_model_1b1r(J,h,T,steps);
target_metrics.Si = mean(target_series,2);
target_metrics.SiSj = (target_series * target_series') ./ steps;
target_metrics.cov = cov(target_series');
target_metrics.FC = corr(target_series','Type','Pearson');

[h_est, J_est, performance] = inverse_ising_fit_with_gradients_Adam(target_metrics, T, steps, 0.1, 0.1, 0.05, 0.05, 1000, true);

%% 基于SC的预定义模型
% T = 1;
% J = readmatrix("..\demo\input\Desikan_input\sc_train.csv");
% N = length(J);
% J(eye(N)==1) = 0;
% J = J./max(J,[],"all");
% 
% h = 0.1*ones([N,1])+0.1*(rand([N,1])-0.5);
% steps = 10000;
% 
% target_series = simulate_Ising_model_1b1r(J,h,T,steps);
% target_metrics.Si = mean(target_series,2);
% target_metrics.SiSj = (target_series * target_series') ./ steps;
% target_metrics.cov = cov(target_series');
% target_metrics.FC = corr(target_series','Type','Pearson');
% 
% [h_est, J_est, performance] = inverse_ising_fit_with_gradients_Adam(target_metrics, T, steps, 0.1, 0.1, 0.05, 0.1, 1000, true);
% % [h_est, J_est, performance] = inverse_ising_fit_with_gradients(target_metrics, T, steps, 1, 0.01, 1000, true);

%% 更大的随机矩阵预定义模型
% T = 1;
% N = 360;
% J = randn(360);
% J(eye(N)==1) = 0;
% h = 0.1*ones([N,1])+0.1*(rand([N,1])-0.5);
% steps = 10000;
% 
% target_series = simulate_Ising_model_1b1r(J,h,T,steps);
% target_metrics.Si = mean(target_series,2);
% target_metrics.SiSj = (target_series * target_series') ./ steps;
% target_metrics.cov = cov(target_series');
% target_metrics.FC = corr(target_series','Type','Pearson');
% 
% [h_est, J_est, performance] = inverse_ising_fit_with_gradients_Adam(target_metrics, T, steps, 0.1, 0.1, 0.05, 0.1, 1000, true);