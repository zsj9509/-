function x = myPCG_X(x,Cha,Mu_x,Mu_y,Gam,M1,M2,beta,dim)
  temp=beta*Cha(:)+Gam(:)+beta*(diff_xT(Mu_x,dim)+diff_yT(Mu_y,dim))-...
  (diff_xT(M1,dim)+diff_yT(M2,dim));                                                
  [x, ~] = pcg(@(x) Fun(x), temp, 1e-4,1000,[],[],x);   
    function y = Fun(x)
         y=beta*x+beta*(diff_xT(diff_x(x,dim),dim)+diff_yT(diff_y(x,dim),dim));
    end
end
