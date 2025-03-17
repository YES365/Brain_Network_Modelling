clear;
load("..\input\connectivity_matrices\subject1_scan1.mat");

% 时间测试标准，单位1ms，时长600s(10min)，测试平台9950X
dt = 0.001;    
T = 600;    

w = 0.8;
I = 0.35;
G = 1.2;
sigma = 0.006;

% 约2.2s
tic
S = dMFM(connectivity_density_matrix,dt,T,w,I,G,sigma);
toc
