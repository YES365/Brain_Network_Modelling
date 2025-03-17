function [E,I] = Wilson_Cowan_d(SC, DM, dt, T, P, varargin)
%WILSON_COWAN_D 有时延的Wilson-Cowan模型模拟
%   必要参数为SC, DM, dt, P, c5
%   SC为结构连接矩阵，P为(n,T)外部刺激序列，dt为模拟的时间步长，c5为全局耦合系数
%   DM为时间延迟矩阵，矩阵元(i,j)为脑区i、j间的时间延迟，单位应为s
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
p.addParameter('c5',1.5);  
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
c5 = p.Results.c5;
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
ttotal = ceil(T/dt);
tpre = ceil(1/dt);
delay = round(DM/dt);
delay_m = max(delay,[],'all');

E = ones([n ttotal+tpre+delay_m])/10;
I = ones([n ttotal+tpre+delay_m])/10;

E_d = zeros(size(delay));

%% 使用每行循环的方式提取延迟矩阵元

% for t = delay_m+1:delay_m+tpre
%     indexing = t - delay;
%     for i=1:n
%         E_d(i,:) = E(i,indexing(i,:));
%     end
%     E(:,t+1) = E(:,t) + dt*(1/tau).*(-E(:,t) + (S_emax-E(:,t)).* ...
%         S_e(c1*E(:,t)-c2*I(:,t)+c5*diag(SC*E_d))) + sigma*randn([n,1])*sqrt(dt);
%     I(:,t+1) = I(:,t) + dt*(1/tau)*(-I(:,t) + (S_imax-I(:,t)).* ...
%         S_i(c3*E(:,t)-c4*I(:,t))) + sigma*randn([n,1])*sqrt(dt);
% end
% 
% for t = delay_m+tpre:length(E)-1
%     indexing = t - delay;
%     for i=1:n
%         E_d(i,:) = E(i,indexing(i,:));
%     end
%     E(:,t+1) = E(:,t) + dt*(1/tau).*(-E(:,t) + (S_emax-E(:,t)).* ...
%         S_e(c1*E(:,t)-c2*I(:,t)+c5*diag(SC*E_d)+P(:,t+1-tpre-delay_m))) + sigma*randn([n,1])*sqrt(dt);
%     I(:,t+1) = I(:,t) + dt*(1/tau)*(-I(:,t) + (S_imax-I(:,t)).* ...
%         S_i(c3*E(:,t)-c4*I(:,t))) + sigma*randn([n,1])*sqrt(dt);
% end

%% 使用sub2ind进行矩阵索引，效率提高约40%
% 时间延迟计算的基本思路是提取E_d矩阵，E_d(i,j) = E_i(t-tau_ij),是第i个区域依照i、j间时延得到的脑区活动值
% 那么 SC*E_d(i,i) = sum_j SC_ij*E_j(t-tau_ji)，就是我们需要的值

dim1 = zeros([n,n]);
for i = 1:n
    dim1(i,:) = i;
end

for t = delay_m+1:delay_m+tpre
    indexing = t - delay;
    sub = sub2ind(size(E),dim1,indexing);
    E_d = E(sub);
    E(:,t+1) = E(:,t) + dt*(1/tau).*(-E(:,t) + (S_emax-E(:,t)).* ...
        S_e(c1*E(:,t)-c2*I(:,t)+ c5*diag(SC*E_d) )) + sigma*randn([n,1])*sqrt(dt);
    I(:,t+1) = I(:,t) + dt*(1/tau)*(-I(:,t) + (S_imax-I(:,t)).* ...
        S_i(c3*E(:,t)-c4*I(:,t))) + sigma*randn([n,1])*sqrt(dt);
end

for t = delay_m+tpre:length(E)-1
    indexing = t - delay;
    sub = sub2ind(size(E),dim1,indexing);
    E_d = E(sub);
    E(:,t+1) = E(:,t) + dt*(1/tau).*(-E(:,t) + (S_emax-E(:,t)).* ...
        S_e(c1*E(:,t)-c2*I(:,t)+ c5*diag(SC*E_d) +P(:,t+1-tpre-delay_m))) + sigma*randn([n,1])*sqrt(dt);
    I(:,t+1) = I(:,t) + dt*(1/tau)*(-I(:,t) + (S_imax-I(:,t)).* ...
        S_i(c3*E(:,t)-c4*I(:,t))) + sigma*randn([n,1])*sqrt(dt);
end

E = E(:,delay_m+tpre+1:end);
I = I(:,delay_m+tpre+1:end);

end

