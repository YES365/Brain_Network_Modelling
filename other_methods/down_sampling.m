function [x_d] = down_sampling(x,dt,TR,pre)
%DOWN_SAMPLING 下采样函数，用于将模拟的时间采样至实测数据的时间尺度
%   x为需要下采样的时间序列（第二维为时间）
%   dt为x的时间精度，TR为目标时间精度，pre为需要舍弃的预运行时长

times = size(x,2);
T = dt*times;
x_d = zeros([size(x,1) length(pre:TR:T)]);

for i=1:length(pre:TR:T)
    x_d(:,i) = x(:,floor((pre+(i-1)*TR)/dt));
end

end

