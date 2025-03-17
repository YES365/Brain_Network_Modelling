function [h_est, J_est, performance] = inverse_ising_fit_with_gradients_Adam(target_metrics, temperature, series_length, learning_rate_h, learning_rate_J, tolorance_h, tolorance_J, max_iteration, print_info)
%INVERSE_ISING_FIT_WITH_GRADIENTS 将Ising模型拟合至目标数据
% 输入参数:
%   target_metrics  - struct, 目标数据的特征，包括<s_i> double(N*1), 
%                   <s_i s_j> double(N*N), cov double(N*N), FC double(N*N)
%   temperature     - double, Ising模型的温度
%   series_length   - int, 每次迭代生成的Ising模型序列长度
%   learning_rate_h   - double, Adam算法的学习率，参数h
%   learning_rate_J   - double, Adam算法的学习率，参数J
%   tolorance_h       - double, 终止条件，如果迭代中window_size步的平均梯度模拟下降小于阈值，则终止迭代
%   tolorance_J       - double, 终止条件，参数J
%   max_iteration   - int, 最大迭代步数
%   print_info      - logical, 是否输出迭代信息
% 输出:
%   h_est           - double(N*1), 最优的h参数
%   J_est           - double(N*N), 最优的J参数
%   performance     - double(4*num_iter), 4个维度依次为FC和<s_i>与目标值的相关系数，
%                       以及grad_J和grad_h的平均2范数

%% 读入目标数据的特征
Si_emp = target_metrics.Si;
SiSj_emp = target_metrics.SiSj;
cov_emp = target_metrics.cov;
FC_emp = target_metrics.FC;

%% 准备初始参数
N = length(Si_emp);
J_est = -inv(cov_emp)./2;
J_est(eye(N)==1) = 0;
J_est = (J_est + J_est') / 2; 
h_est = Si_emp;

learning_rate_J = learning_rate_J*(10^floor(log10(abs(mean(abs(J_est),"all")))));
learning_rate_h = learning_rate_h*(10^floor(log10(abs(mean(abs(h_est),"all")))));
% learning_rate_J = 0.001;
% learning_rate_h = 0.001;
tolorance_J = tolorance_J*(10^floor(log10(abs(mean(abs(J_est),"all")))));
tolorance_h = tolorance_h*(10^floor(log10(abs(mean(abs(h_est),"all")))));

% Adam参数
beta1 = 0.9;               % 一阶矩估计的指数衰减率
beta2 = 0.999;             % 二阶矩估计的指数衰减率
epsilon = 1e-8;            % 防止除零的小常数

m_h = zeros(size(h_est));   % 一阶矩估计
v_h = zeros(size(h_est));   % 二阶矩估计
m_J = zeros(size(J_est));   % 一阶矩估计
v_J = zeros(size(J_est));   % 二阶矩估计

%% 基于Adam算法的Ising模型优化
num_iter = 0;
window_size = 50;
h_est_history = cell([1, window_size]);
J_est_history = cell([1, window_size]);
norm_grad_h_history = zeros([1, window_size]);
norm_grad_J_history = zeros([1, window_size]);
norm_grad_h_mean = inf;
norm_grad_J_mean = inf;
window_count = 0;
while (num_iter < max_iteration) 
    num_iter = num_iter + 1;
    window_count = window_count + 1;
    tic
    % Ising模型序列生成
    spins = simulate_Ising_model_1b1r(J_est, h_est, temperature, series_length);
    % 模拟特征提取
    Si_sim = mean(spins,2);
    SiSj_sim = (spins * spins') ./ series_length;
    % 梯度计算
    grad_h = Si_emp - Si_sim;
    grad_J = SiSj_emp - SiSj_sim;
    % 更新一阶矩估计
    m_h = beta1 .* m_h + (1 - beta1) .* grad_h;
    m_J = beta1 .* m_J + (1 - beta1) .* grad_J;
    % 更新二阶矩估计
    v_h = beta2 .* v_h + (1 - beta2) .* (grad_h.^2);
    v_J = beta2 .* v_J + (1 - beta2) .* (grad_J.^2);
    % 计算偏差修正
    m_h_hat = m_h / (1 - beta1^num_iter);
    v_h_hat = v_h / (1 - beta2^num_iter);
    m_J_hat = m_J / (1 - beta1^num_iter);
    v_J_hat = v_J / (1 - beta2^num_iter);
    % 梯度下降
    h_est = h_est + learning_rate_h * m_h_hat ./ (sqrt(v_h_hat) + epsilon);
    J_est = J_est + learning_rate_J * m_J_hat ./ (sqrt(v_J_hat) + epsilon);
    % 评估梯度范数
    norm_grad_h = sqrt(norm(grad_h)./N);
    norm_grad_J = sqrt(norm(grad_J,'fro')./(N^2));
    cost_time = toc;
    if window_count > window_size
        norm_grad_h_mean_now = mean(norm_grad_h_history);
        norm_grad_J_mean_now = mean(norm_grad_J_history);
        learning_rate_J = learning_rate_J * 0.1;
        learning_rate_h = learning_rate_h * 0.1;
        if (norm_grad_h_mean - norm_grad_h_mean_now > tolorance_h) || (norm_grad_J_mean - norm_grad_J_mean_now > tolorance_J)
            window_count = 0;
            norm_grad_h_mean = norm_grad_h_mean_now;
            norm_grad_J_mean = norm_grad_J_mean_now;
        else
            fprintf('Early stopping at iteration %d: No significant improvement in gradient norms\n', num_iter);
            break
        end
    else
        norm_grad_h_history(window_count) = norm_grad_h;
        norm_grad_J_history(window_count) = norm_grad_J;
        h_est_history{window_count} = h_est;
        J_est_history{window_count} = J_est;
    end
    if print_info
        FC_sim = corr(spins','Type','Pearson');
        mask = triu(true(N),1);
        FC_cc = corr(FC_emp(mask),FC_sim(mask),"Type","Pearson");
        Si_cc = corr(Si_emp,Si_sim,"Type","Pearson");
        fprintf(['for iteration %d, cost_time is %.2f, norm_grad_h is %.6f, norm_grad_J is %.6f, ' ...
            'FC correlation is %.3f, <s> correlation is %.3f, mean_grad_h is %.6f, mean_grad_J is %.6f\n'], ...
            num_iter, cost_time, norm_grad_h, norm_grad_J, FC_cc, Si_cc, mean(grad_h), mean(grad_J,"all"))
        performance(:,num_iter) = [FC_cc;Si_cc;norm_grad_J;norm_grad_h];
    end
end

h_rank = tiedrank(norm_grad_h_history);
J_rank = tiedrank(norm_grad_J_history);
totol_rank = h_rank + J_rank;
[~, opt_index] = min(totol_rank);
h_est = h_est_history{opt_index};
J_est = J_est_history{opt_index};

if ~print_info
    FC_sim = corr(spins','Type','Pearson');
    mask = triu(true(N),1);
    performance(1) = corr(FC_emp(mask),FC_sim(mask),"Type","Pearson");
    performance(2) = corr(Si_emp,Si_sim,"Type","Pearson");
    performance(3) = num_iter;
    fprintf('Iteration results: FC correlation is %.3f, <s> correlation is %.3f\n', performance(1), performance(2))
end

end

