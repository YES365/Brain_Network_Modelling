% 比较不同迭代方案的模拟速度，并检验模拟结果
% 经典的随机选取N次节点的迭代方法是最终能量最低的
clear

%% 定义模拟系统
T = 1;
N = 360;
J = randn(N);
% J = readmatrix("C:\Users\YES365\OneDrive\My_Files\My_Study\My_Research\Doctoral_Projects\critial_parietal_lobe\input\Desikan_input\sc_train.csv");
% J(eye(N)==1) = 0;
% J = J./max(J,[],"all");
% J = readmatrix("C:\Users\77293\OneDrive\My_Files\My_Study\My_codes\Brain_Network_Modelling\demo\input\Desikan_input\fc_train.csv");
% J = -inv(J)./2;
% J(eye(N)==1) = 0;
h = 0.1*ones([N,1]);

steps = 10000;

tic
spins_1b1r = simulate_Ising_model_1b1r(J,h,T,steps);
toc
tic
spins_1b1 = simulate_Ising_model_1b1(J,h,T,steps);
toc
tic
spins_1b1rs = simulate_Ising_model_1b1rs(J,h,T,steps);
toc

E_1b1r = Ising_energy(J,h,spins_1b1r);
E_1b1 = Ising_energy(J,h,spins_1b1);
E_1b1rs = Ising_energy(J,h,spins_1b1rs);

% 假设 `energies` 是记录的能量
beta = 1 / T; % 逆温度
figure;

energies = E_1b1r;
[hist_counts, energy_bins] = histcounts(energies, 'Normalization', 'pdf'); % 模拟分布
energy_centers = (energy_bins(1:end-1) + energy_bins(2:end)) / 2;

% 理论分布（玻尔兹曼分布）
boltzmann_pdf = exp(-beta * energy_centers);
boltzmann_pdf = boltzmann_pdf / sum(boltzmann_pdf); % 归一化

% 绘图
subplot(1,3,1)
bar(energy_centers, hist_counts, 'FaceAlpha', 0.5); % 模拟分布
% hold on;
% plot(energy_centers, boltzmann_pdf, 'r-', 'LineWidth', 2); % 理论分布
legend('模拟分布', '理论玻尔兹曼分布');
xlabel('能量 E');
ylabel('概率密度 P(E)');
title('能量分布比较 1b1r');

energies = E_1b1;
[hist_counts, energy_bins] = histcounts(energies, 'Normalization', 'pdf'); % 模拟分布
energy_centers = (energy_bins(1:end-1) + energy_bins(2:end)) / 2;

% 理论分布（玻尔兹曼分布）
boltzmann_pdf = exp(-beta * (energy_centers - max(energy_centers)));
boltzmann_pdf = boltzmann_pdf / sum(boltzmann_pdf); % 归一化

% 绘图
subplot(1,3,2)
bar(energy_centers, hist_counts, 'FaceAlpha', 0.5); % 模拟分布
% hold on;
% plot(energy_centers, boltzmann_pdf, 'r-', 'LineWidth', 2); % 理论分布
legend('模拟分布', '理论玻尔兹曼分布');
xlabel('能量 E');
ylabel('概率密度 P(E)');
title('能量分布比较 1b1');

energies = E_1b1rs;
[hist_counts, energy_bins] = histcounts(energies, 'Normalization', 'pdf'); % 模拟分布
energy_centers = (energy_bins(1:end-1) + energy_bins(2:end)) / 2;

% 理论分布（玻尔兹曼分布）
boltzmann_pdf = exp(-beta * energy_centers);
boltzmann_pdf = boltzmann_pdf / sum(boltzmann_pdf); % 归一化

% 绘图
subplot(1,3,3)
bar(energy_centers, hist_counts, 'FaceAlpha', 0.5); % 模拟分布
% hold on;
% plot(energy_centers, boltzmann_pdf, 'r-', 'LineWidth', 2); % 理论分布
legend('模拟分布', '理论玻尔兹曼分布');
xlabel('能量 E');
ylabel('概率密度 P(E)');
title('能量分布比较 1b1rs');

cov_1b1rs = cov(spins_1b1rs');
cov_1b1 = cov(spins_1b1');
cov_1b1r = cov(spins_1b1r');

% cov_1b1rs = spins_1b1rs*spins_1b1rs';
% cov_1b1 = spins_1b1*spins_1b1';
% cov_1b1r = spins_1b1r*spins_1b1r';

figure(2)
subplot(2,2,1)
heatmap(J)
colormap turbo
grid off
subplot(2,2,2)
heatmap(cov_1b1r)
colormap turbo
grid off
subplot(2,2,3)
heatmap(cov_1b1)
colormap turbo
grid off
subplot(2,2,4)
heatmap(cov_1b1rs)
colormap turbo
grid off

mask = triu(true(N),1);
corr(cov_1b1(mask),J(mask))
corr(cov_1b1r(mask),J(mask))
corr(cov_1b1rs(mask),J(mask))

J_inv = inv(J);
cov_1b1_inv = -inv(cov_1b1)./2;
cov_1b1_inv(eye(N)==1) = 0;
cov_1b1r_inv = -inv(cov_1b1r)./2;
cov_1b1r_inv(eye(N)==1) = 0;
cov_1b1rs_inv = -inv(cov_1b1rs)./2;
cov_1b1rs_inv(eye(N)==1) = 0;

corr(cov_1b1_inv(mask),J(mask))
corr(cov_1b1r_inv(mask),J(mask))
corr(cov_1b1rs_inv(mask),J(mask))

figure(3)
subplot(2,2,1)
heatmap(J)
colormap turbo
grid off
subplot(2,2,2)
heatmap(cov_1b1r_inv)
colormap turbo
grid off
subplot(2,2,3)
heatmap(cov_1b1_inv)
colormap turbo
grid off
subplot(2,2,4)
heatmap(cov_1b1rs_inv)
colormap turbo
grid off
