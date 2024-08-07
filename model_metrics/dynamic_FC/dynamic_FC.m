function [dFC] = dynamic_FC(BOLD, width, is_continue)
%DYNAMIC_FC 计算动态FC
%   BOLD 为各脑区的BOLD信号, n*t           
%   width 为dFC滑动窗口的宽度
%   is_continue为1时，窗口间隔为1，is_continue为0时，窗口间隔为width

t = size(BOLD,2);
n = size(BOLD,1);

if is_continue == 1
    dFC = zeros([n,n,(t-width+1)]);
    for i=1:(t-width+1)
        fc = corrcoef(BOLD(:,i:i+width-1)');
        dFC(:,:,i) = fc;
    end
else
    dFC = zeros([n,n,length(1:width:t)]);
    j=1;
    for i=1:width:t
        fc = corrcoef(BOLD(:,i:i+width-1)');
        dFC(:,:,j) = fc;
        j = j+1;
    end
end

end

