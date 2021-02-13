clear ;
clc;	

addpath(genpath('TV_operator'))
addpath(genpath('quality assess'))
addpath(genpath('Common'))

%% simulated experiment 1
% ----------------------------load image-----------------------------------
load simu_indian
Ohsi       = simu_indian;
noiselevel = 0.1*ones(1,224); 

% ------------------------ Simulation experiment --------------------------
Nhsi      = Ohsi;
[M,N,p]   = size(Ohsi);
%% Gaussian noise
for i = 1:p
     Nhsi(:,:,i)=Ohsi(:,:,i)+noiselevel(i)*randn(M,N);
end

%% TV sparsity denoising
rank   = [13,13,13];
pp=2;

for i=1:p
    bandp = Ohsi(:,:,i);
    Q(:,i)= bandp(:);
end

W=repmat(Q(:,[1:14,80:93,143:156,200:213]),[1,4]);
c = 80; % num of atoms
t = 40;
params.data=W;
params.Tdata=t;
params.dictsize=c;
params.memusage='high';
[X,~,~]=ksvd(params,'');
X=X./repmat(sqrt(sum(X.^2,1)),size(X,1),1);
X=orth(X);
F1=X;
F2=eye(224);
clear X Y params c t W

tau    = 0.004 *sqrt(M*N);
[ output_image,output_X,output_L] = TVF(Nhsi,tau,rank,pp,F1,F2);
[mpsnr,mssim,ergas]=msqia(Ohsi,output_image)
