function [E] = Wilson_Cowan(SC, dt, P, c5, varargin)
%WILSON_COWAN 无时延的Wilson-Cowan模型模拟
%   必要参数为SC, dt, P, c5
%   SC为结构连接矩阵，P为(n,T)外部刺激序列，dt为模拟的时间步长，c5为全局耦合系数
%   返回值为兴奋性神经活动 E 的(n,T)时间序列
%
%   可选参数为tau, c1, c2, c3, c4, sigma, a_e, a_i, theta_e, theta_i
%   tau, c1, c2, c3, c4, sigma为参数，其中tau为时间常数，sigma为噪声标准差
%   a_e, a_i, theta_e, theta_i为转移函数参数
%   默认值参考：Muldoon, Sarah Feldt, et al. "Stimulation-based control of 
%   dynamic brain networks." PLoS computational biology 12.9 (2016): e1005076.
%   https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005076

p = inputParser;            % 函数的输入解析器
p.addParameter('c1',16);      % 设置变量名和默认参数
p.addParameter('c2',12);      
p.addParameter('c3',15);      
p.addParameter('c4',3);      
p.addParameter('a_e',1.3);      
p.addParameter('theta_e',4);      
p.addParameter('a_i',2);     
p.addParameter('theta_i',3.7);      
p.addParameter('tau',0.008);      % 8ms
p.addParameter('sigma',0.00001);      
parse(p,varargin{:});       % 对输入变量进行解析，如果检测到前面的变量被赋值，则更新变量取值 

%% 读入参数
c1 = p.Results.c1;
c2 = p.Results.c2;
c3 = p.Results.c3;
c4 = p.Results.c4;
a_e = p.Results.a_e;
theta_e = p.Results.theta_e;
a_i = p.Results.a_i;
theta_i = p.Results.theta_i;
tau = p.Results.tau;    
sigma = p.Results.sigma;

n = length(SC); %节点数量

S_e = @(x)WC_transfer_function(x, a_e, theta_e);
S_i = @(x)WC_transfer_function(x, a_i, theta_i);

S_emax = WC_S_max(a_e, theta_e);
S_imax = WC_S_max(a_i, theta_i);
%% 进行模拟
T = size(P, 2);
% tpre = ceil(1/dt);

% E = ones([n T+tpre])/10;
% I = ones([n T+tpre])/10;

E = ones([n T])/10;
I = ones([n T])/10;

% % 预运行1s
% for t = 1:tpre-1
%     E(:,t+1) = E(:,t) + dt*(1/tau).*(-E(:,t) + (S_emax-E(:,t)).* ...
%         S_e(c1*E(:,t)-c2*I(:,t)+c5*SC*E(:,t)+P(:,t+1))) + sigma*randn([n,1])*sqrt(dt);
%     I(:,t+1) = I(:,t) + dt*(1/tau)*(-I(:,t) + (S_imax-I(:,t)).* ...
%         S_i(c3*E(:,t)-c4*I(:,t))) + sigma*randn([n,1])*sqrt(dt);
% end


for t = 1:length(E)-1
    E(:,t+1) = E(:,t) + dt*(1/tau).*(-E(:,t) + (S_emax-E(:,t)).* ...
        S_e(c1*E(:,t)-c2*I(:,t)+c5*SC*E(:,t)+P(:,t-tpre+1))) + sigma*randn([n,1])*sqrt(dt);
    I(:,t+1) = I(:,t) + dt*(1/tau)*(-I(:,t) + (S_imax-I(:,t)).* ...
        S_i(c3*E(:,t)-c4*I(:,t))) + sigma*randn([n,1])*sqrt(dt);
end

end

