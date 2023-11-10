% 2维参数搜索样例代码
% 使用EI_dMFM模型，仅使用CPU进行计算
% 使用ADM 5950x，16线程并行，耗时5765s

clear;

%% 读取数据
% SC数据是建模必要的，FC则作为评价指标
% SC一般是纤维数加权矩阵，且通过归一化控制数值
% FC一般不考虑负连接，提取FC的上三角部分用于计算相关系数
SC = readmatrix("input\Desikan_input\sc_train.csv");
FC = readmatrix("input\Desikan_input\fc_train.csv");
SC = SC ./ max(SC,[],"all");
FC(eye(size(FC))==1) = 0;
FC(FC < 0) = 0;

n = length(SC); % 节点数量

mask = zeros(n);
for i=1:n
    for j=i+1:n
        mask(i,j) = 1;
    end
end
mask = mask == 1;
mask = mask(:);

FC = FC(mask);

%% 设置模型参数

% 设置激活函数
H_e = @(x)dMFM_H(x,310,125,0.16);
H_i = @(x)dMFM_H(x,615,177,0.087);

% 设置参数
tau = zeros([n 2]);
tau(:,1) = 0.1; % tau_e = 0.1s
tau(:,2) = 0.01; % tau_i = 0.01s

sigma = 0.01;
I = 0;

% 模拟10min的fMRI扫描
dt = 0.001;
T = 660;
TR = 0.7;
pre = 60;

%% 进行搜索

% 定义要搜索的参数及范围
w_ie_range = 0.5:0.05:3.0;
G_range = 0.5:0.05:3.0;

cc_boldi_map = zeros([length(w_ie_range) length(G_range)]);

L1 = length(w_ie_range);
L2 = length(G_range);

% 模型是噪声驱动的，具有随机性，对于某些指标，需要重复模拟才能得到可靠的结果
repeat = 1;

% 纯CPU计算使用parfor并行不同参数下的模型
tic
for i = 1:L1
    parfor j = 1:L2
    w = ones([n 5]);
    w(:,1) = 1; % w_e = 1.0
    w(:,2) = 0.7; % w_i = 1.0
    w(:,3) = 1.4; % w_ee = 1.4
    w(:,4) = 1; % w_ei = 1
    w(:,5) = w_ie_range(i);

    G = G_range(j);

    disp([G,w_ie_range(i)])

    tic
    FC_sim = zeros([n n repeat]);
    for r=1:repeat
        % 此处使用I_E作为血氧模型的输入
        [~, I_E, ~, ~] = EI_dMFM(SC, dt, T, w, I, G, sigma, H_e, H_i, tau);
        
        BOLD = Balloon_Windkessel_model(I_E,dt);
        BOLD = down_sampling(BOLD,dt,TR,pre);
    
        FC_sim(:,:,r) = corrcoef(BOLD');
    end
    FC_sim = mean(FC_sim,3);
    FC_sim(eye(size(FC_sim))==1) = 0;
    FC_sim(FC_sim < 0) = 0;

    FC_sim = FC_sim(mask);

    cc = corrcoef(FC,FC_sim);
    cc_boldi_map(i,j) = cc(1,2);
    toc
    end
end
toc

fig = figure(1);
heatmap(G_range,w_ie_range,cc_boldi_map)
colormap turbo
xlabel('G')
ylabel('w_{ie}')
title('cc BOLD I_E')
saveas(fig, "Figures\EI_dMFM_2d_para_search_CPU", 'png');

save("output\EI_dMFM_2d_para_search_CPU","cc_boldi_map")