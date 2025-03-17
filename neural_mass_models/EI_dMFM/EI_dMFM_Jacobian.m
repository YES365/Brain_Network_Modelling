function [Jacob, S_E_star] = EI_dMFM_Jacobian(SC, w, I, G, H_E, H_I, tau, varargin)
%EI_DMFM_JACOBIAN 此处显示有关此函数的摘要
%   此处显示详细说明
N = length(SC);
p = inputParser;            % 函数的输入解析器
p.addParameter('J',0.15); % nA
p.addParameter('I_b',0.382);    % nA
p.addParameter('gamma',0.641);
p.addParameter('S_E_0',rand([N 1]));
parse(p,varargin{:}); 

J = p.Results.J;
I_b = p.Results.I_b;
gamma = p.Results.gamma; 

dH_E = @(x) - 310./(exp(20 - (248.*x)./5) - 1) - ...
    (248.*exp(20 - (248.*x)./5).*(310.*x - 125))./(5*(exp(20 - (248.*x)./5) - 1).^2);
dH_I = @(x) - 615./(exp(15399/1000 - (10701.*x)./200) - 1) - ...
    (10701.*exp(15399/1000 - (10701.*x)./200).*(615.*x - 177))./(200*(exp(15399/1000 - (10701.*x)./200) - 1).^2);

dt = 0.001;
T = 60;
sigma = 0;

w_ee = w(:,3);
w_ei = w(:,4);
w_ie = w(:,5);

[S_E, I_E, ~, I_I] = EI_dMFM(SC, dt, T, w, I, G, sigma, H_E, H_I, tau, 'J', J, 'I_b', I_b, 'gamma', gamma, 'S_E_0', p.Results.S_E_0);
S_E_star = S_E(:,end);
x_E_star = I_E(:,end);
x_I_star = I_I(:,end);

Jacob_EE = zeros(N);
Jacob_EI = zeros(N);
Jacob_IE = zeros(N);
Jacob_II = zeros(N);

for i = 1:N
    for j = 1:N
        if i == j
            Jacob_EE(i,j) = -1/tau(i,1)-gamma*H_E(x_E_star(i))+w_ee(i)*gamma*J*(1-S_E_star(i))*dH_E(x_E_star(i));
            Jacob_EI(i,j) = w_ei(i)*dH_I(x_I_star(i));
            Jacob_IE(i,j) = -w_ie(i)*J*(1-S_E_star(i))*gamma*dH_E(x_E_star(i));
            Jacob_II(i,j) = -1/tau(i,2) - dH_I(x_I_star(i));
        else
            Jacob_EE(i,j) = gamma*G*J*(1-S_E_star(i))*SC(i,j)*dH_E(x_E_star(i));
        end
    end
end

Jacob = zeros(2*N); 
Jacob(1:N,1:N) = Jacob_EE;
Jacob(1:N,N+1:2*N) = Jacob_EI;
Jacob(N+1:2*N,1:N) = Jacob_IE;
Jacob(N+1:2*N,N+1:2*N) = Jacob_II;

end

