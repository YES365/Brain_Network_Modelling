function [BOLD] = Balloon_Windkessel_model(s,dt)
%BALLOON_WINDKESSEL_MODEL 血氧动力学的Balloon-Windkessel model
%   s(n,t)为神经活动的时间序列, dt为时间精度
%   Balloon-Windkessel model将神经活动转化为BOLD信号
%   参数参考：https://www.sciencedirect.com/science/article/pii/S0896627319300443

va_shape = size(s);
x = zeros(va_shape);
f = ones(va_shape);
v = ones(va_shape);
q = ones(va_shape);

%% parameters from Reference
epsilon = 100;
kappa = 0.65;   % 1/s
gamma = 0.41;   % 1/s
tau = 0.98;     % s
alpha = 0.32;
rho = 0.34;

k1 = 3.72;
k2 = 0.53;
k3 = 0.53;

%% parameters from DTB
% epsilon = 200;
% kappa = 1/0.8;
% gamma = 1/0.4;
% alpha = 0.2;
% tau = 1;
% rho = 0.8;

% k1 = 7*rho;
% k2 = 2;
% k3 = 2*rho - 0.2;


V0 = 0.02;
alpha_div = 1./alpha;

for i = 1:va_shape(2)-1
    % 血流动力学演化
    x(:,i+1) = x(:,i) + (epsilon*s(:,i) - kappa.*x(:,i) - gamma.*(f(:,i)-1)).*dt;
    f(:,i+1) = f(:,i) + (x(:,i)).*dt;
    v(:,i+1) = v(:,i) + 1/tau.*(f(:,i)-v(:,i).^alpha_div).*dt;
    q(:,i+1) = q(:,i) + ...
        1/tau.*( f(:,i)./rho.*(1-(1-rho).^(1./f(:,i)))-q(:,i)./v(:,i).*v(:,i).^alpha_div)*dt;
end

% 数据转换为BOLD信号

BOLD = V0*(k1*(1-q) + k2*(1-q./v) + k3*(1-v)); 

end

