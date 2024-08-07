function conn = array2network(array, have_diagonal)
%ARRAY2NETWORK 将对称矩阵的上三角拉伸为的向量折叠回矩阵
%   have_diagonal 是否包含对角线
a = length(array);
if have_diagonal
    n = (sqrt(8*a+1) - 1)/2;
    conn = zeros(n);
    k=1;
    for i=1:n
        for j=i:n
            conn(i,j) = array(k);
            conn(j,i) = array(k);
            k=k+1;
        end
    end
else
    n = (sqrt(8*a+1) + 1)/2;
    conn = zeros(n);
    k=1;
    for i=1:n
        for j=i+1:n
            conn(i,j) = array(k);
            conn(j,i) = array(k);
            k=k+1;
        end
    end
end

end

