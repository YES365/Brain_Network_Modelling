function [S_E, I_E, S_I, I_I, S_E_end, S_I_end] = EI_dMFM_gpu(SC, dt, T, w, I, G, sigma, H_e, H_i, tau, S_E_0, S_I_0, varargin)
%EI_DMFM 有E-I平衡的二维动态平均场网络模型
%   SC为结构连接矩阵
%   dt为时间间隔，T是模拟的时间长度
%   w为区域内连接强度，按顺序共有w_e, w_i, w_ee, w_ei, w_ie五个维度，即(n,5)
%   I为外部输入，G为全局耦合强度,sigma为噪声尺度
%   dt为时间精度，T为模拟时长

%% 读入参数
% 预设参数
n = length(SC); % 节点数量

p = inputParser;            % 函数的输入解析器
p.addParameter('J',0.15); % nA
p.addParameter('I_b',0.382);    % nA
p.addParameter('gamma',0.641);
parse(p,varargin{:}); 

J = p.Results.J;
I_b = p.Results.I_b;
gamma = p.Results.gamma; 


% 设置突触时间尺度
tau_e = tau(:,1);
tau_i = tau(:,2);

% 设置突触权重
w_e = w(:,1);
w_i = w(:,2);
w_ee = w(:,3);
w_ei = w(:,4);
w_ie = w(:,5);

%% 进行模拟
% 设置模拟时长
tpost = length(0:dt:T); % 实际运行 Ts 

S_E = gpuArray(zeros([n, tpost]));
S_I = gpuArray(zeros([n, tpost]));
S_E(:,1) = S_E_0;
S_I(:,1) = S_I_0;

I_E = gpuArray(zeros([n, tpost]));
I_I = gpuArray(zeros([n, tpost]));

% EI_dMFM方程迭代
for t=1:tpost-1
     I_E(:,t+1) = w_e*I_b + (w_ee.*S_E(:,t) + G.*(SC*S_E(:,t)))*J - w_ie.*S_I(:,t) + I;
     I_I(:,t+1) = w_i*I_b + w_ei.*S_E(:,t)*J - S_I(:,t);

     S_E(:,t+1) = S_E(:,t) + dt*(-S_E(:,t)./tau_e + ...
         gamma*(1-S_E(:,t)).*H_e(I_E(:,t+1))) + sigma*randn([n 1],"double",'gpuArray')*sqrt(dt);
     S_I(:,t+1) = S_I(:,t) + dt*(-S_I(:,t)./tau_i + ...
         H_i(I_I(:,t+1))) + sigma*randn([n 1],"double",'gpuArray')*sqrt(dt);
%      S_E(S_E(:,t+1)<0,t+1) = 0;
%      S_E(S_E(:,t+1)>1,t+1) = 1;
%      S_I(S_I(:,t+1)<0,t+1) = 0;
%      S_I(S_I(:,t+1)>1,t+1) = 1;
end

S_E_end = S_E(:,end);
S_I_end = S_I(:,end);

end

