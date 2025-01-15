function [dTS] = dynamic_time_scale(time_series, width, lag_max, is_continue, TR)
%DYNAMIC_TIME_SCALE 计算滑动窗口的时间尺度（即ACF的最大连续显著阶数）
%   time_series 为目标时间序列, n*t           
%   width 为dFC滑动窗口的宽度, lag_max 为最大检验阶数
%   is_continue为1时，窗口间隔为1，is_continue为0时，窗口间隔为width

t = size(time_series,2);
n = size(time_series,1);

if is_continue == 1
    dTS = zeros([n,(t-width+1)]);
    for i=1:(t-width+1)
       bold = time_series(:,i:i+width-1);
       for k=1:n
           data = zscore(bold(k,:));
           k_max = ACF_largest_k(data, lag_max, 0.05, 1);
           dTS(k,i) = k_max*TR;
       end
    end
else
    dTS = zeros([n,length(1:width:t)]);
    j=1;
    for i=1:width:t
        bold = time_series(:,i:i+width-1);
       for k=1:n
           data = zscore(bold(k,:));
           k_max = ACF_largest_k(data, lag_max, 0.05, 1);
           dTS(k,j) = k_max*TR;
       end
        j = j+1;
    end
end


end

