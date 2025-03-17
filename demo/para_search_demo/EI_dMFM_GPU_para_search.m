% 2维参数搜索样例代码
% 使用EI_dMFM模型，仅使用CPU进行计算
% 使用3080ti 12G，ensemble上限 > 模型数，仅运行一轮
% 用时4000s左右

clear;

%% 读取数据
% SC数据是建模必要的，FC则作为评价指标
% SC一般是纤维数加权矩阵，且通过归一化控制数值
% FC一般不考虑负连接，提取FC的上三角部分用于计算相关系数
SC = readmatrix("..\input\Desikan_input\sc_train.csv");
FC = readmatrix("..\input\Desikan_input\fc_train.csv");
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

%% 设置标量参数

% 设置激活函数
H_e = @(x)dMFM_H(x,310,125,0.16);
H_i = @(x)dMFM_H(x,615,177,0.087);

sigma = 0.01;
I = 0;

% 模拟10min的fMRI扫描
dt = 0.001;
T = 660;
TR = 0.7;
pre = 60;

observe = ceil((T-pre)/TR);
pre = ceil(pre/TR);

%% 需要搜索的参数

w_ie_range = 0.5:0.05:3.0;
G_range = 0.5:0.05:3.0;

[w_ie, G] = meshgrid(w_ie_range, G_range);
w_ie = w_ie(:);
G = G(:);
num_paras = length(w_ie);

cc_boldi = zeros([num_paras 1]);

cc_boldi_map = zeros([length(w_ie_range) length(G_range)]);

tic
%% 分配至ensemble模型进行模拟
% ensemble的最大值依赖于显存
ensemble = cast(5000,'int32');          % 单GPU最大并行模型数
round = idivide(num_paras, ensemble);   % 完整模拟组数
left = mod(num_paras, ensemble);        % 不完整模拟参数数量

%% 完整
for r=1:round
    
    % 定义数组参数，使用GPU数组
    w = ones([n 5]);
    w(:,1) = 1; % w_e = 1.0
    w(:,2) = 0.7; % w_i = 1.0
    w(:,3) = 1.4; % w_ee = 1.4
    w(:,4) = 1; % w_ei = 1
    
    tau = zeros([n 2]);
    tau(:,1) = 0.1; % tau_e = 0.1s
    tau(:,2) = 0.01; % tau_i = 0.01s
    
    % 生成ensemble参数
    w_en = gpuArray(ones([n*ensemble 5]));
    tau_en = gpuArray(zeros([n*ensemble 2]));
    G_en = gpuArray(zeros([n*ensemble 1]));
    
    for i=1:ensemble
        w(:,5) = w_ie(ensemble*(r-1) + i);
        w_en(((i-1)*n+1):(i*n),:) = w;
        tau_en(((i-1)*n+1):(i*n),:) = tau;
        G_en(((i-1)*n+1):(i*n),:) = G(ensemble*(r-1) + i);
    end
    
    % 用稀疏矩阵生成ensemble结构连接
    x = zeros([n*n*ensemble 1]);
    y = zeros([n*n*ensemble 1]);
    v = zeros([n*n*ensemble 1]);
    count = 1;
    for k=1:ensemble
        for i=1:n
            for j=1:n
                x(count) = (k-1)*n+i;
                y(count) = (k-1)*n+j;
                v(count) = SC(i,j);
                count = count+1;
            end
        end
    end
    SC_en = sparse(x,y,v);
    SC_en = gpuArray(SC_en);
    
    bold = zeros([ensemble*n observe]);
    
    % 预运行
    S_E_0 = rand([ensemble*n 1]);
    S_I_0 = rand([ensemble*n 1]);
    x_0 = 0;
    f_0 = 1;
    v_0 = 1;
    q_0 = 1;
    for i=1:pre
        tic
        [~, I_E, ~, ~,S_E_0,S_I_0] = EI_dMFM_gpu(SC_en, dt, TR, w_en, I, G_en, sigma, H_e, H_i, tau_en,S_E_0,S_I_0);
        [~,x_0,f_0,v_0,q_0] = Balloon_Windkessel_model_gpu(I_E,dt,x_0,f_0,v_0,q_0);
        toc
    end
    
    % 正式运行
    for i=1:observe
        disp(['observe ',num2str(i)])
        tic
        [~, I_E, ~, ~,S_E_0,S_I_0] = EI_dMFM_gpu(SC_en, dt, TR, w_en, I, G_en, sigma, H_e, H_i, tau_en,S_E_0,S_I_0);
        [bold(:,i),x_0,f_0,v_0,q_0] = Balloon_Windkessel_model_gpu(I_E,dt,x_0,f_0,v_0,q_0);
        toc
    end
    
    % 计算指标
    tic
    for i=1:ensemble
        BOLD = bold(((i-1)*n+1):(i*n),:);
        FC_sim = corrcoef(BOLD');
        FC_sim(eye(size(FC_sim))==1) = 0;
        FC_sim(FC_sim < 0) = 0;
    
        FC_sim = FC_sim(mask);
    
        cc = corrcoef(FC,FC_sim);
        cc_boldi(ensemble*(r-1) + i) = cc(1,2);
    end
    toc

end

%% 不完整模拟
ensemble = left;
r = round + 1;
% 定义数组参数，使用GPU数组
w = ones([n 5]);
w(:,1) = 1; % w_e = 1.0
w(:,2) = 0.7; % w_i = 1.0
w(:,3) = 1.4; % w_ee = 1.4
w(:,4) = 1; % w_ei = 1

tau = zeros([n 2]);
tau(:,1) = 0.1; % tau_e = 0.1s
tau(:,2) = 0.01; % tau_i = 0.01s

% 生成ensemble参数
w_en = gpuArray(ones([n*ensemble 5]));
tau_en = gpuArray(zeros([n*ensemble 2]));
G_en = gpuArray(zeros([n*ensemble 1]));

for i=1:ensemble
    w(:,5) = w_ie(ensemble*(r-1) + i);
    w_en(((i-1)*n+1):(i*n),:) = w;
    tau_en(((i-1)*n+1):(i*n),:) = tau;
    G_en(((i-1)*n+1):(i*n),:) = G(ensemble*(r-1) + i);
end

% 用稀疏矩阵生成ensemble结构连接
x = zeros([n*n*ensemble 1]);
y = zeros([n*n*ensemble 1]);
v = zeros([n*n*ensemble 1]);
count = 1;
for k=1:ensemble
    for i=1:n
        for j=1:n
            x(count) = (k-1)*n+i;
            y(count) = (k-1)*n+j;
            v(count) = SC(i,j);
            count = count+1;
        end
    end
end
SC_en = sparse(x,y,v);
SC_en = gpuArray(SC_en);

bold = zeros([ensemble*n observe]);

% 预运行
S_E_0 = rand([ensemble*n 1]);
S_I_0 = rand([ensemble*n 1]);
x_0 = 0;
f_0 = 1;
v_0 = 1;
q_0 = 1;
for i=1:pre
    tic
    [~, I_E, ~, ~,S_E_0,S_I_0] = EI_dMFM_gpu(SC_en, dt, TR, w_en, I, G_en, sigma, H_e, H_i, tau_en,S_E_0,S_I_0);
    [~,x_0,f_0,v_0,q_0] = Balloon_Windkessel_model_gpu(I_E,dt,x_0,f_0,v_0,q_0);
    toc
end

% 正式运行
for i=1:observe
    disp(['observe ',num2str(i)])
    tic
    [~, I_E, ~, ~,S_E_0,S_I_0] = EI_dMFM_gpu(SC_en, dt, TR, w_en, I, G_en, sigma, H_e, H_i, tau_en,S_E_0,S_I_0);
    [bold(:,i),x_0,f_0,v_0,q_0] = Balloon_Windkessel_model_gpu(I_E,dt,x_0,f_0,v_0,q_0);
    toc
end

% 计算指标
tic
for i=1:ensemble
    BOLD = bold(((i-1)*n+1):(i*n),:);
    FC_sim = corrcoef(BOLD');
    FC_sim(eye(size(FC_sim))==1) = 0;
    FC_sim(FC_sim < 0) = 0;

    FC_sim = FC_sim(mask);

    cc = corrcoef(FC,FC_sim);
    cc_boldi(ensemble*(r-1) + i) = cc(1,2);
end
toc

%% 整理结果
toc

for i=1:num_paras
    cc_boldi_map(w_ie_range==w_ie(i), G_range==G(i)) = cc_boldi(i);
end

fig = figure(1);
heatmap(G_range,w_ie_range,cc_boldi_map)
colormap turbo
xlabel('G')
ylabel('w_{ie}')
title('cc BOLD I_E')
saveas(fig, "..\Figures\EI_dMFM_2d_para_search_GPU", 'png');

save("..\output\EI_dMFM_2d_para_search_GPU","cc_boldi_map")