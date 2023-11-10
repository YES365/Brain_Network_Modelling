function [variability] = node_variability(BOLD, width, is_continue)
%BRAIN_VARIABILITY 计算节点动态可变性
%   BOLD 为各脑区的BOLD信号, n*t           
%   width 为dFC滑动窗口的宽度
%   is_continue为1时，窗口间隔为1，is_continue为0时，窗口间隔为width

t = size(BOLD,2);
n = size(BOLD,1);

if is_continue == 1
    dFC = zeros([(t-width+1) n n]);
    for i=1:(t-width+1)
        dFC(i,:,:) = corrcoef(BOLD(:,i:i+width-1)');
    end
else
    dFC = zeros([length(1:width:t) n n]);
    j=1;
    for i=1:width:t
        dFC(j,:,:) = corrcoef(BOLD(:,i:i+width-1)');
        j = j+1;
    end
end

variability = zeros([n 1]);

for i=1:n
    dFC_node = squeeze(dFC(:,:,i));
    cc = corrcoef(dFC_node');
    variability(i) = 1 - mean(triu(cc),"all");
end

end

