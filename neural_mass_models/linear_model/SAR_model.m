function y = SAR_model(SC, dt, T, k, sigma, y_0)
%SAR_model 一个线性的自回归模型
%   SC 结构连接矩阵，k 全局耦合强度，sigma噪声标准差
%   SAR_model数学形式参考：
%   https://www.sciencedirect.com/science/article/pii/S1053811915000932

n = length(SC);
y = zeros([n,length(0:dt:T)]);
y(:,1) = y_0;

for t=1:length(0:dt:T)-1
    y(:,t+1) = k*SC*y(:,t) + sigma.*randn([n 1]).*sqrt(dt);
end

end