function [S] = WC_transfer_function(x, a, theta)
%TRANSFER_FUNCTION 
%   Wilson-Cowan模型的转移函数/激活函数
%   是一个平移的sigmoid函数，由a,theta两个参数，x为自变量
S = 1./(1+exp(-a*(x-theta))) - 1./(1+exp(a*theta));
end

