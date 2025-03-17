function [h_est, J_est, performance] = inverse_ising_fit_with_gradients(target_metrics, temperature, series_length, learning_rate_h, learning_rate_J, tolorance_h, tolorance_J, max_iteration, print_info)
%INVERSE_ISING_FIT_WITH_GRADIENTS 将Ising模型拟合至目标数据
% 输入参数:
%   target_metrics  - struct, 目标数据的特征，包括<s_i> double(N*1), 
%                   <s_i s_j> double(N*N), cov double(N*N), FC double(N*N)
%   temperature     - double, Ising模型的温度
%   series_length   - int, 每次迭代生成的Ising模型序列长度
%   learning_rate   - double, 梯度下降的学习率
%   tolorance       - double, 终止条件，梯度的模值
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
% h_est = atanh(Si_emp) - J_est * Si_emp;
h_est = zeros([N,1]);

%% 梯度下降
% 尝试J和h交替更新的方法
% 以连续3次变换符号为震荡标志,检测到震荡则缩小学习率
% 尝试每十次切换优化目标
norm_grad_h = inf;
norm_grad_J = inf;
flag_updata_J = false;
sign_grad_J = 10*ones([1,4]);
sign_grad_h = 10*ones([1,4]);
decrease_rate = 0.1;
% decay_rate = 0.5;
num_iter = 0;
iter_count = 0;
while (num_iter < max_iteration) && ((norm_grad_h > tolorance_h)||(norm_grad_J > tolorance_J))
    num_iter = num_iter + 1;
    iter_count = iter_count + 1; 
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
    if flag_updata_J
        J_est = J_est + learning_rate_J * grad_J;
    else
        h_est = h_est + learning_rate_h * grad_h;
    end
    % 评估梯度
    norm_grad_h = sqrt(norm(grad_h)./N);
    norm_grad_J = sqrt(norm(grad_J,'fro')./(N^2));
    mean_grad_h = mean(grad_h);
    mean_grad_J = mean(grad_J,"all");
    cost_time = toc;
    if print_info
        FC_sim = corr(spins','Type','Pearson');
        mask = triu(true(N),1);
        FC_cc = corr(FC_emp(mask),FC_sim(mask),"Type","Pearson");
        Si_cc = corr(Si_emp,Si_sim,"Type","Pearson");
        fprintf(['for iteration %d, cost_time is %.2f, norm_grad_h is %.6f, norm_grad_J is %.6f, ' ...
            'FC correlation is %.3f, <s> correlation is %.3f, mean_grad, mean_grad_h is %.6f, mean_grad_J is %.6f\n'], ...
            num_iter, cost_time, norm_grad_h, norm_grad_J, FC_cc, Si_cc, mean_grad_h, mean_grad_J)
        performance(:,num_iter) = [FC_cc;Si_cc;norm_grad_J;norm_grad_h];
    end
    if flag_updata_J
        sign_grad_J = [sign(mean_grad_J), sign_grad_J(1:3)];
        oscillation_sign = sum(sign_grad_J);
        if oscillation_sign == 0
            fprintf('grad J sign is [%d, %d, %d, %d], oscillation occur, learning rate J decrease, switch to h\n', sign_grad_J)
            flag_updata_J = false;
            learning_rate_J = decrease_rate*learning_rate_J;
            sign_grad_J = 10*ones([1,4]);
            iter_count = 0;
        end
    else
        sign_grad_h = [sign(mean_grad_h), sign_grad_h(1:3)];
        oscillation_sign = sum(sign_grad_h);
        if oscillation_sign == 0
            fprintf('grad h sign is [%d, %d, %d, %d], oscillation occur, learning rate h decrease, switch to J\n', sign_grad_h)
            flag_updata_J = true;
            learning_rate_h = decrease_rate*learning_rate_h;
            sign_grad_h = 10*ones([1,4]);
            iter_count = 0;
        end
    end
    if iter_count == 8
        iter_count = 0;
        flag_updata_J = ~flag_updata_J;
        sign_grad_J = 10*ones([1,4]);
        sign_grad_h = 10*ones([1,4]);
        fprintf('switch side %d\n', flag_updata_J)
        % if flag_updata_J
        %     learning_rate_h = decay_rate*learning_rate_h;
        % else
        %     learning_rate_J = decay_rate*learning_rate_J;
        % end
    end
end

if ~print_info
    FC_sim = corr(spins','Type','Pearson');
    mask = triu(true(N),1);
    performance(1) = corr(FC_emp(mask),FC_sim(mask),"Type","Pearson");
    performance(2) = corr(Si_emp,Si_sim,"Type","Pearson");
    performance(3) = num_iter;
    fprintf('Iteration results: FC correlation is %.3f, <s> correlation is %.3f\n', performance(1), performance(2))
end

end

