clear;
load("input\connectivity_matrices\subject1_scan1.mat");

%% 单模型测试

% 时间测试标准，单位0.1ms，时长600s(10min)
dt = 0.001;    
T = 600;    

n = length(connectivity_density_matrix); %节点数量

% 设置激活函数
% 需要频繁迭代的函数不建议设置varargin
H_e = @(x)dMFM_H(x,310,125,0.16);
H_i = @(x)dMFM_H(x,615,177,0.087);

% 设置参数
tau = zeros([n 2]);

tau(:,1) = 0.1; % tau_e = 0.1s
tau(:,2) = 0.01; % tau_i = 0.01s

sigma = 0.01;
I = 0;

w = ones([n 5]);
w(:,2) = 0.7; % w_i = 1.0
w(:,3) = 1.4; % w_ee = 1.4

G = 1.5;

% 约64s
tic
[S_E, I_E, S_I, I_I] = EI_dMFM(connectivity_density_matrix, dt, T, w, I, G, sigma, H_e, H_i, tau);
toc

%% 多模型并行测试

% % 时间测试标准，单位0.1ms，时长2.2s
% dt = 0.001;    
% T = 2.2;    
% 
% n = length(connectivity_density_matrix); %节点数量
% 
% % 设置激活函数
% % 需要频繁迭代的函数不建议设置varargin
% H_e = @(x)dMFM_H(x,310,125,0.16);
% H_i = @(x)dMFM_H(x,615,177,0.087);
% 
% % 设置参数
% tau = zeros([n 2]);
% 
% tau(:,1) = 0.1; % tau_e = 0.1s
% tau(:,2) = 0.01; % tau_i = 0.01s
% 
% sigma = 0.01;
% I = 0;
% 
% w = ones([n 5]);
% w(:,2) = 0.7; % w_i = 1.0
% w(:,3) = 1.4; % w_ee = 1.4
% 
% G = 1.5;
% 
% ensemble = 2000;
% 
% % 约32s
% bold = zeros([n ensemble]);
% tic
% parfor i=1:ensemble
%     [S_E{i}, ~, ~, ~] = EI_dMFM(connectivity_density_matrix, dt, T, w, I, G, sigma, H_e, H_i, tau);
% %     BOLD = Balloon_Windkessel_model(S_E,dt);
% %     bold(:,i) = BOLD(:,end);
% end
% toc