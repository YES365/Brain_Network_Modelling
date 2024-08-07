function array = network2array(conn, have_diagonal)
%NETWORK2ARRAY 将对称矩阵的上三角拉伸为向量
%   have_diagonal 是否包含对角线

n = length(conn);
if have_diagonal
    array = zeros([1 n*(n+1)/2]);
    k=1;
    for i=1:n
        for j=i:n
            array(k) = conn(i,j);
            k=k+1;
        end
    end
else
    array = zeros([1 n*(n-1)/2]);
    k=1;
    for i=1:n
        for j=i+1:n
            array(k) = conn(i,j);
            k=k+1;
        end
    end
end

end

