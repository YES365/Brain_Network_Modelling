function S = WC_S_max(a,theta)
%S_MAX 得到S_max的值
%   有a,theta两个参数
S = 1 - 1/(1+exp(a*theta));
end

