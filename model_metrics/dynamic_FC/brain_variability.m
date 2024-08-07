function [variability] = brain_variability(BOLD, width, is_continue)
%BRAIN_VARIABILITY 计算全脑动态可变性
%   BOLD 为各脑区的BOLD信号, n*t           
%   width 为dFC滑动窗口的宽度
%   is_continue为1时，窗口间隔为1，is_continue为0时，窗口间隔为width
%   关于动态可变性指标与其生理意义，可以参考：
%   https://academic.oup.com/brain/article/139/8/2307/1753964

t = size(BOLD,2);
n = size(BOLD,1);

mask = zeros(n);
for i=1:n
    for j=i+1:n
        mask(i,j) = 1;
    end
end
mask = mask == 1;
mask = mask(:);

if is_continue == 1
    dFC = zeros([sum(mask) (t-width+1)]);
    for i=1:(t-width+1)
        fc = corrcoef(BOLD(:,i:i+width-1)');
        dFC(:,i) = fc(mask);
    end
else
    dFC = zeros([sum(mask) length(1:width:t)]);
    j=1;
    for i=1:width:t
        fc = corrcoef(BOLD(:,i:i+width-1)');
        dFC(:,j) = fc(mask);
        j = j+1;
    end
end

cc = corrcoef(dFC);
n = size(cc,1);
mask = zeros(n);
for i=1:n
    for j=i+1:n
        mask(i,j) = 1;
    end
end
mask = mask == 1;
mask = mask(:);

variability = 1 - mean(cc(mask));
end