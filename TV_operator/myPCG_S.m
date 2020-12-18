function x = myPCG_S(x,S1,S2,H1,H2,beta,dim)
 temp=+beta*(diff_xT(S1,dim)+diff_yT(S2,dim))-(diff_xT(H1,dim)+diff_yT(H2,dim));                                                
  [x, ~] = pcg(@(x) Fun(x), temp, 1e-4,1000,[],[],x);   
    function y = Fun(x)
         y=eps+beta*(diff_xT(diff_x(x,dim),dim)+diff_yT(diff_y(x,dim),dim));
    end
end