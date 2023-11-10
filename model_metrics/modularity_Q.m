function [Q] = modularity_Q(A,c)
%modularity_Q 计算网络的模块度Q
%   采用Rubinov and Sporns (2011)提出的非对称负连接计算方式
%   A为n*n邻接矩阵，c为n*1模块划分向量，数值为1:C

C = length(unique(c));
A(eye(size(A)) == 1) = 0;

% N = length(A);
% mask = zeros(N);
% for i=1:N
%     for j=i+1:N
%         mask(i,j) = 1;
%     end
% end
% mask = mask == 1;
% mask = mask(:);

A_p = A;
A_p(A_p < 0) = 0;

A_n = A;
A_n(A_n > 0) = 0;
A_n = -A_n;

Q_p = 0;
v_p = sum(A_p,"all");
% v_p = sum(A_p(mask));
s_p = sum(A_p,2);
for k=1:C
    net = A_p(c==k,c==k);
    s = s_p(c==k);
    n = length(net);
    q = 0;
    for i=1:n
        for j=1:n
            if ~(i==j)
                q = q+(net(i,j) - s(i)*s(j)/v_p);
            end
        end
    end
    Q_p = Q_p + q;
end

Q_n = 0;
v_n = sum(A_n,"all");
% v_n = sum(A_n(mask));
s_n = sum(A_n,2);
for k=1:C
    net = A_n(c==k,c==k);
    s = s_n(c==k);
    n = length(net);
    q = 0;
    for i=1:n
        for j=1:n
            if ~(i==j)
                q = q+(net(i,j) - s(i)*s(j)/v_n);
            end
        end
    end
    Q_n = Q_n + q;
end

Q = Q_p/v_p - Q_n/(v_p+v_n);

end