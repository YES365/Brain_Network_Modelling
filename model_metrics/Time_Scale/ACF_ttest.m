function [p, t] = ACF_ttest(acf,N,m)
%ACF_TTEST 计算ACF的显著性
%   acf 被检验的ACF数值，N 时间序列长度，m 需要检验ACF阶数
%   计算方法参考：Analysis of Financial Time Series, Third Edition, 
%   Ruey S. Tsay, section 2.2

t = zeros([m,1]);
p = zeros([m,1]);
for i=1:m
    t(i) = acf(i) / sqrt((1 + 2*sum(acf(1:i).^2)) / N);
    p(i) = tcdf(t(i),N-i,"upper");
end

end

