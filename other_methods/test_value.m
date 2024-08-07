function [z_score] = test_value(array,value_to_test,z_threshold)
%TTEST_VALUE 用zscore法判断一个数值大于给定数组代表的分布
%   数组array和要检验的值value_to_test，zscore阈值z_threshold 

% 计算数组的统计量
array_mean = mean(array);
array_std = std(array);

% Z-score方法
z_score = (value_to_test - array_mean) / array_std;

fprintf('Z-score方法\n');
fprintf('Z-score: %.4f\n', z_score);
if z_score > z_threshold
    fprintf('结论: 值 %.2f 可能不属于该分布 (|Z| > %.2f)\n\n', value_to_test, z_threshold);
else
    fprintf('结论: 没有足够证据表明值 %.2f 不属于该分布 (|Z| <= %.2f)\n\n', value_to_test, z_threshold);
end

end

