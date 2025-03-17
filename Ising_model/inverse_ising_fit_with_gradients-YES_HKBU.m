function [h_est, J_est, performance] = inverse_ising_fit_with_gradients(target_metrics, temperature, learning_rate_J, learning_rate_h, tolorance, max_iteration, series_length, print_info)
%INVERSE_ISING_FIT_WITH_GRADIENTS 将Ising模型拟合至目标数据
% 输入参数:
%   target_metrics  - struct, 目标数据的特征，包括<s_i> double(N*1), 
%                   <s_i s_j> double(N*N), cov double(N*N), FC double(N*N)
%   temperature     - double, Ising模型的温度
%   learning_rate   - double, 梯度下降的学习率
%   tolorance       - double, 终止条件，梯度的模值
%   max_iteration   - int, 最大迭代步数
%   series_length   - int, 每次迭代生成的Ising模型序列长度
%   print_info      - logical, 是否输出迭代信息
% 输出:
%   h_est           - double(N*1), 最优的h参数
%   J_est           - double(N*N), 最优的J参数
%   performance     - double(1*2), 最优参数的FC和<s_i>与目标值的相关系数

%% 读入目标数据的特征
Si_emp = target_metrics.Si;
SiSj_emp = target_metrics.SiSj;
cov_emp = target_metrics.cov;
FC_emp = target_metrics.FC;

%% 准备初始参数
N = length(Si_emp);
h_est = zeros([N,1]);
J_est = -inv(cov_emp)./2;
J_est(eye(N)==1) = 0;
J_est = (J_est + J_est') / 2; 

norm_grad = inf;
num_iter = 0;
%% 梯度下降
while (num_iter < max_iteration) && (norm_grad > tolorance)
    tic
    % Ising模型序列生成
    spins = simulate_Ising_model_1b1r(J_est, h_est, temperature, series_length);
    % 模拟特征提取
    Si_sim = mean(spins,2);
    SiSj_sim = (spins * spins') ./ series_length;
    % 梯度计算
    grad_h = Si_emp - Si_sim;
    grad_J = SiSj_emp - SiSj_sim;
    % 梯度下降
    h_est = h_est + learning_rate_h * grad_h;
    J_est = J_est + learning_rate_J * grad_J;

    num_iter = num_iter + 1;
    norm_grad = norm(grad_J) + norm(grad_h);
    cost_time = toc;
    if print_info
        % norm_grad_h = norm(grad_h);
        % norm_grad_J = norm(grad_J);
        % fprintf('for iteration %d, cost_time is %.2f, norm_grad_h is %.6f, norm_grad_J is %.6f\n', num_iter, cost_time, norm_grad_h,norm_grad_J)
        mean_grad_h = mean(grad_h);
        mean_grad_J = mean(grad_J,"all");
        mean_h = mean(h_est);
        mean_J = mean(J_est,"all");
        fprintf('for iteration %d, cost_time is %.2f, mean_grad_h is %.6f, mean_h is %.6f, mean_grad_J is %.6f, mean_J is %.6f\n', num_iter, cost_time, mean_grad_h, mean_h, mean_grad_J, mean_J)
        FC_sim = corr(spins','Type','Pearson');
        mask = triu(true(N),1);
        performance(1) = corr(FC_emp(mask),FC_sim(mask),"Type","Pearson");
        performance(2) = corr(Si_emp,Si_sim,"Type","Pearson");
        fprintf('Iteration results: FC correlation is %.3f, <s> correlation is %.3f\n', performance(1), performance(2))
    end
end

FC_sim = corr(spins','Type','Pearson');
mask = triu(true(N),1);
performance(1) = corr(FC_emp(mask),FC_sim(mask),"Type","Pearson");
performance(2) = corr(Si_emp,Si_sim,"Type","Pearson");
fprintf('Iteration results: FC correlation is %.3f, <s> correlation is %.3f\n', performance(1), performance(2))

end

