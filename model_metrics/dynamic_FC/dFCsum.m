function [dFC_sum] = dFCsum(dFC)
%DFCSUM 计算dFC的和
%  dFC 大小n*n*T

T = size(dFC,3);

dFC_sum = zeros([1,T]);
for i=1:T
    FC_i = dFC(:,:,i);
    dFC_sum(i) = sum(FC_i,"all");
end

end

