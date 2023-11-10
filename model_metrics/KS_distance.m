function [distance] = KS_distance(cdf_1, cdf_2)
%KS_DISTANCE 计算两个累计分布的最大垂直距离
%   cdf_1和cdf_2是两个累计分布函数的值，从0到1，应该是长度相等的向量

if (length(cdf_1) == length(cdf_2)) && (max(cdf_1) == 1) && ...
        (max(cdf_2) == 1) && (min(cdf_1) == 0) && (min(cdf_2) == 0)
    distance = max(abs(cdf_1 - cdf_2));
else
    error('cdf_1 and cdf_2 are not matched cumulative distribution functions')
end

end