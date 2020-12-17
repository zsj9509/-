function [ output_image,output_X,output_L,U_x,V_x] =TVF(oriData3_noise,tau,r,pp,F1,F2)
tol     = 1e-6;
maxIter = 50;
rho     = 1.5;
max_mu  = 1e6;
mu      = 1e-2;
au     = 1/tau; 
lambda  = 1/tau; 
[M,N,p] = size(oriData3_noise);
sizeD   = size(oriData3_noise);
D       = zeros(M*N,p) ;
for i=1:p
    bandp = oriData3_noise(:,:,i);
    D(:,i)= bandp(:);
end
normD   = norm(D,'fro');
%% FFT setting
h               = sizeD(1);
w               = sizeD(2);
d               = sizeD(3);
%% 
Eny_x   = ( abs(psf2otf([+1; -1], [h,w,d])) ).^2  ;
Eny_y   = ( abs(psf2otf([+1, -1], [h,w,d])) ).^2  ;
Eny_z   = ( abs(psf2otf([+1, -1], [w,d,h])) ).^2  ;
Eny_z   =  permute(Eny_z, [3, 1 2]);
determ  =  Eny_x + Eny_y + Eny_z;
%% Initializing optimization variables
X              = randn(M*N,p);
E              = zeros(M*N,p);
L              = D-X-E;
H              = F1'*L*F2;
% U_x and V_x initial
tv_x           = diff_x(X,sizeD);
tv_x           = reshape(tv_x,[M*N,p]);
[U_x,S_x,V_x]  = svd(tv_x,'econ');
U_x            = U_x(:,1:r(1))*S_x(1:r(1),1:r(1));
V_x            = V_x(:,1:r(1));
% U_y and V_y initial
tv_y           = diff_y(X,sizeD);
tv_y           = reshape(tv_y,[M*N,p]);
[U_y,S_y,V_y]  = svd(tv_y,'econ');
U_y            = U_y(:,1:r(2))*S_y(1:r(2),1:r(2));
V_y            = V_y(:,1:r(2));
% tv_z initial
tv_z           = diff_z(X,sizeD);
tv_z           = reshape(tv_z,[M*N,p]);
[U_z,S_z,V_z]  = svd(tv_z,'econ');
U_z            = U_z(:,1:r(3))*S_z(1:r(3),1:r(3));
V_z            = V_z(:,1:r(3));
M1 =zeros(size(D));  % multiplier for D-X-L-E
M2 =zeros(size(D));  % multiplier for Dx_X-U_x*V_x
M3 =zeros(size(D));  % multiplier for Dy_X-U_y*V_y
M4 =zeros(size(D));  % multiplier for Dz_X-U_z*V_z
M5 =zeros(size(D));  % multiplier for L=F1*H*F2'---1
% main loop
iter = 0;
tic
while iter<maxIter
    iter          = iter + 1;   
    Mu_x          = U_x*V_x';
    Mu_y          = U_y*V_y';
    Mu_z          = U_z*V_z';
    diffT_p  = diff_xT(mu*Mu_x-M2,sizeD);
    diffT_p  = diffT_p+diff_yT(mu*Mu_y-M3,sizeD)+diff_zT(mu*Mu_z-M4,sizeD);
    numer1   = reshape( diffT_p + mu*(D(:)-L(:)-E(:)) + M1(:), sizeD);
    x        = real( ifftn( fftn(numer1) ./ (mu*determ + mu) ) );
    X        = reshape(x,[M*N,p]);
    %% -Update U_x and U_y and U_z
    tmp_x         = reshape(diff_x(X,sizeD),[M*N,p]);
    tmp_x         = tmp_x+M2/mu;
    
    
    U_x           = softthre(tmp_x*V_x, 1/mu);    
    tmp_y         = reshape(diff_y(X,sizeD),[M*N,p]);
    tmp_y         = tmp_y+M3/mu;
    
    
    U_y           = softthre(tmp_y*V_y, 1/mu);
    tmp_z         = reshape(diff_z(X,sizeD),[M*N,p]);
    tmp_z         = tmp_z+M4/mu;
    
    
    U_z           = softthre(tmp_z*V_z, 1/mu); 
    
    %% -Update V_x and V_y and V_z
    [u,~,v]       = svd(tmp_x'*U_x,'econ');
    V_x           = u*v';
    [u,~,v]       = svd(tmp_y'*U_y,'econ');
    V_y           = u*v';
    [u,~,v]       = svd(tmp_z'*U_z,'econ');
    V_z           = u*v';
    %% -Update L 
    L = (D -X -E +F1*H*F2' +(M1 -M5)/mu)/2;
    %% -Update H 
    H = F1'*svdthresh(L + M5/mu, au/mu)*F2;
    %% -Update E 
    if (pp==1)
        E             = softthre(D-X-L+M1/mu, lambda/mu);
    else
        E_c=D-X-L+M1/mu;
        E             = shrinkage_Lp(E_c(:),pp,lambda,mu);
        E =reshape(E,[h*w,d]);
    end
    
    %% stop criterion  
    leq1 = D -X -L -E;
    leq2 = reshape(diff_x(X,sizeD),[M*N,p])- U_x*V_x';
    leq3 = reshape(diff_y(X,sizeD),[M*N,p])- U_y*V_y';
    leq4 = reshape(diff_z(X,sizeD),[M*N,p])- U_z*V_z';
    leq5 = L - F1*H*F2';
    stopC1 = norm(leq1,'fro')/normD;
    stopC2 = max(abs(leq2(:)));
    stopC4 = norm(leq4,'fro')/normD;
    stopC5 = norm(leq5,'fro')/normD;
    disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e')  ...
            ',Y-X-L-E=' num2str(stopC1,'%2.3e') ',||DX-UV||=' num2str(stopC2,'%2.3e')...
            ',|DZ-UV|' num2str(stopC4,'%2.3e')]);
    if stopC1<tol || stopC2<tol
        break;
    else
        M1 = M1 + mu*leq1;
        M2 = M2 + mu*leq2;
        M3 = M3 + mu*leq3;
        M4 = M4 + mu*leq4;
        M5 = M5 + mu*leq5;
        mu = min(max_mu,mu*rho); 
    end 
end
output_image = reshape(X+L,[M,N,p]);
output_X = reshape(X,[M,N,p]);
output_L = reshape(L,[M,N,p]);
end