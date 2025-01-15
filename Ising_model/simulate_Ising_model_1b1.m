function [spins] = simulate_Ising_model_1b1(J, h, T, steps)
%SIMULATE_ISING_MODEL_1B1 模拟不均匀网络上的Ising模型，每个 MC 步遍历 N 个节点
% 输入参数:
%   J      - 相互作用矩阵 (N x N)，J(i,j) 表示节点 i 和 j 的相互作用强度
%   h      - 外部磁场向量 (N x 1)，h(i) 表示作用在节点 i 上的外部磁场
%   T      - 温度
%   steps  - 模拟的蒙特卡洛步数
% 输出:
%   spins  - 最终的自旋序列 (N x steps)

%   N      - 节点数量（总自旋数）
N = length(h);

% 初始化自旋状态（随机取 -1 或 1），并预迭代100步
spins_now= randi([0, 1], [N,1]) * 2 - 1; % 随机生成 -1 或 1
init_steps = 1000;
for step = 1:init_steps
    for i = 1:N % 每个 MC 步包含 N 次尝试
        % 计算该自旋的局部场（包括邻居的相互作用和外部磁场）
        local_field = J(i, :) * spins_now + h(i);

        % 计算翻转该自旋的能量变化
        dE = 2 * spins_now(i) * local_field;

        % 根据 Metropolis准则决定是否翻转
        if dE <= 0 || rand() < exp(-dE ./ T)
            spins_now(i) = -spins_now(i); % 翻转自旋
        end
    end
end

spins = zeros([N, steps]);

% 蒙特卡洛模拟（Metropolis算法）
for step = 1:steps
    for i = 1:N % 每个 MC 步遍历 N 个节点
        % 计算该自旋的局部场（包括邻居的相互作用和外部磁场）
        local_field = sum(J(i, :) .* spins_now') + h(i);

        % 计算翻转该自旋的能量变化
        dE = 2 * spins_now(i) * local_field;

        % 根据 Metropolis准则决定是否翻转
        if dE <= 0 || rand() < exp(-dE ./ T)
            spins_now(i) = -spins_now(i); % 翻转自旋
        end
    end
    spins(:,step) = spins_now;
end

end

