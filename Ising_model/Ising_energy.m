function [E] = Ising_energy(J, h, spins)
%ISING_ENERGY 计算Ising模型的能量
% 输入参数:
%   J      - 相互作用矩阵 (N x N)，J(i,j) 表示节点 i 和 j 的相互作用强度
%   h      - 外部磁场向量 (N x 1)，h(i) 表示作用在节点 i 上的外部磁场
%   spins  - 自旋序列 (N x T)
% 输出:
%   E      - 能量 (1 x T)
E = zeros([1,size(spins,2)]);
for i=1:size(spins,2)
    E(i) = -spins(:,i)'*(J*spins(:,i) + h);
end

end

