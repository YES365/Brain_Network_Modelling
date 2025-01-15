function [TS] = static_time_scale(time_series,  lag_max, TR)
%STATIC_TIME_SCALE 计算目标时间序列（如BOLD信号）的时间尺度（最大ACF显著阶数）
%   此处显示详细说明
n = size(time_series,1);

TS = zeros([n,1]);
for k=1:n
   data = zscore(time_series(k,:));
   k_max = ACF_largest_k(data, lag_max, 0.05, 1);
   TS(k) = k_max*TR;
end

end

