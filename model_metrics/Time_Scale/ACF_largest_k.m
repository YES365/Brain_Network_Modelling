function [k_max] = ACF_largest_k(data, lag_max, alpha, is_fdr)
%ACF_LARGEST_K 计算时间序列的自回归函数连续显著的最大阶数
%   data(1,T) 为需要检验的一维时间序列
%   lag_max 为ACF计算的最大阶数
%   alpha 为误差限，is_fdr为检验是否使用fdr校正
%   fdr校正程序来自：Víctor Martínez-Cagigal (2024). 
%   Multiple Testing Toolbox 
%   (https://www.mathworks.com/matlabcentral/fileexchange/70604-multiple-testing-toolbox), 
%   MATLAB Central File Exchange. 

[autocorr, ~] = xcorr(data, lag_max, 'normalized');
index = lag_max+1;
% lags = lags(index:end);
[p, ~] = ACF_ttest(autocorr(index:end),length(data),lag_max);
if is_fdr
    p = fdr_BH(p,alpha);     
end

if all(p < alpha)
    k_max = lag_max;
else
    index = find(p > alpha, 1, 'first'); 
    k_max = index - 1;
end

end

