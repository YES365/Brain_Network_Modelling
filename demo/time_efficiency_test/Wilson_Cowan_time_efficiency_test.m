clear;
load("..\input\connectivity_matrices\subject1_scan1.mat");

% 时间测试标准，单位1ms，时长600s(10min)，测试平台9950X
dt = 0.001;    
T = 600;    

n = length(connectivity_density_matrix); %节点数量
tmax = ceil(T/dt);   %运行时步

P = zeros([n,tmax]);
delay_matrix = delay_matrix./1000;  %单位统一至s

c5 = 1.5;

% 约3.2s
tic
[E,I] = Wilson_Cowan(connectivity_density_matrix, dt, T, P);
toc

% 约38s
tic
[E,I] = Wilson_Cowan_d(connectivity_density_matrix, delay_matrix, dt, T, P);
toc
