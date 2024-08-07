function [S] = dMFM(SC, dt, T, w, I, G, sigma, varargin)
%dMFM dMFM模型的网络模拟
%   SC为结构连接矩阵
%   dt为时间间隔，T是模拟的时间长度
%   w为自反馈连接强度，I为外部输入，G为全局耦合强度,sigma为噪声尺度
%   关于 dMFM 的提出，可以参考：
%   https://www.jneurosci.org/content/33/27/11239

p = inputParser;            % 函数的输入解析器
p.addParameter('J',0.2609); % nA
p.addParameter('tau_s',0.1);    % s 
p.addParameter('gamma_s',0.641);
parse(p,varargin{:}); 

J = p.Results.J;
tau_s = p.Results.tau_s;
gamma_s = p.Results.gamma_s; 

a = 270; %n/C (/nC?)
b = 108; % Hz
d = 0.154; % s
H = @(x)dMFM_H(x,a,b,d);

n = length(SC); % 节点数量
% tpre = ceil(10/dt);  % 预运行10s
% if dt <= 0.001
%     tpre = ceil(1/dt);  % 预运行1s
% end
% tpost = ceil(T/dt); % 实际运行 Ts 

ttotal = ceil(T/dt); % 实际运行 Ts 
S = ones([n, ttotal])/10;

for t=1:ttotal-1
     x = w.*J.*S(:,t) + G.*J.*SC*S(:,t) + I;
     S(:,t+1) = S(:,t) + dt.*(-S(:,t)./tau_s + ...
         gamma_s.*(1-S(:,t)).*H(x)) + sigma.*randn([n 1]).*sqrt(dt);
end

% S = S(:, tpre+1:end);

end

