function [crossRatio] = dCrossHemisphere(dFC)
%DCROSSHEMISPHERE 计算dFC的半球内连接与半球间连接比例
%  dFC 大小n*n*T，脑区排列为左右分开的顺序

T = size(dFC,3);
n = size(dFC,2);

crossRatio = zeros([1,T]);
for i=1:T
    FC_i = dFC(:,:,i);
    crossRatio(i) = sum(FC_i(1:(n/2),(n/2+1):n),"all")*2 / ...
        (sum(FC_i(1:(n/2),1:(n/2)),"all") + sum(FC_i((n/2+1):n,(n/2+1):n),"all"));
end

end

