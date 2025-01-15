function [ACF] = static_ACF1(BOLD)
%STATIC_ACF1 计算BOLD信号的一阶自回归函数
%   此处显示详细说明
n = size(BOLD,1);

ACF = zeros([n, 1]);

for k=1:n
   data = zscore(BOLD(k,:));
   [acf,~] = xcorr(data, 1, 'normalized');
   ACF(k) = acf(end);
end

end

