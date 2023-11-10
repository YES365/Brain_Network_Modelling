function [H] = dMFM_H(x, a, b, d)
%DMFM_H dMFM模型的群体发放率函数
%   输入x为群体电流，输出H为群体发放率
%   默认参数参考：
%   https://www.jneurosci.org/content/33/27/11239

H = (a*x - b) ./ (1 - exp(-d*(a*x-b)));

end

