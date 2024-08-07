function [dACF] = dynamic_ACF1(BOLD, width, is_continue)
%DYNAMIC_TIME_SCALE 计算滑动窗口的一阶自回归函数
%   BOLD 为各脑区的BOLD信号, n*t           
%   width 为dFC滑动窗口的宽度
%   is_continue为1时，窗口间隔为1，is_continue为0时，窗口间隔为width

t = size(BOLD,2);
n = size(BOLD,1);

if is_continue == 1
    dACF = zeros([n,(t-width+1)]);
    for i=1:(t-width+1)
       bold = BOLD(:,i:i+width-1);
       for k=1:n
           data = zscore(bold(k,:));
           [acf,~] = xcorr(data, 1, 'normalized');
           dACF(k,i) = acf(end);
       end
    end
else
    dACF = zeros([n,length(1:width:t)]);
    j=1;
    for i=1:width:t
        bold = BOLD(:,i:i+width-1);
       for k=1:n
           data = zscore(bold(k,:));
           [acf,~] = xcorr(data, 1, 'normalized');
           dACF(k,j) = acf(end);
       end
        j = j+1;
    end
end


end

