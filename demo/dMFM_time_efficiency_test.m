clear;
load("input\connectivity_matrices\subject1_scan1.mat");

% 时间测试标准，单位0.1ms，时长600s(10min)
dt = 0.0001;    
T = 0:dt:600;    

w = 0.8;
I = 0.35;
G = 1.2;
sigma = 0.006;

% 约360s
tic
S = dMFM(connectivity_density_matrix,dt,T,w,I,G,sigma);
toc
