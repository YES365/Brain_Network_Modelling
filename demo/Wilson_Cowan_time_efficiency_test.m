clear;
load("input\connectivity_matrices\subject1_scan1.mat");

% 时间测试标准，单位0.1ms，时长600s(10min)
dt = 0.0001;    
T = 0:dt:600;    

n = length(connectivity_density_matrix); %节点数量
tmax = length(T);   %运行时步

P = zeros([n,tmax]);
delay_matrix = delay_matrix./1000;  %单位统一至s

c5 = 1.5;

% 约62s
tic
E = Wilson_Cowan(connectivity_density_matrix, dt, P, c5);
toc

% 约460s
tic
E = Wilson_Cowan_d(connectivity_density_matrix, delay_matrix, dt, P, c5);
toc
