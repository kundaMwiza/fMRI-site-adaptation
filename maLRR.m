function [Z,Ez,Ew,W,Wi] = maLRR(T,S,Dim,Maxiter,alaph,beta)
    
mu = 1e-5;tol = 1e-8;max_mu = 1e6;rho =1.2;iter = 0;
[d,m] = size(T);M = length(S);Z = cell(1,M);Wi = cell(1,M);
W = eye(Dim,d);Ew = cell(1,M);Ez = cell(1,M);F = cell(1,M);Y1 = cell(1,M);
Y2 = cell(1,M);Y3 = zeros(Dim,d);Y4 = cell(1,M);Y1_tmp = cell(1,M);
Y2_tmp = cell(1,M);Y3_tmp =  zeros(Dim,d);Y4_tmp = cell(1,M);stopC = zeros(1,M);J = zeros(d);
for i=1:M
    Z{1,i} = zeros(m,size(S{1,i},2));
    Wi{1,i} = zeros(Dim,d);Ew{1,i} = zeros(Dim,d);
    Ez{1,i} = zeros(Dim,size(S{1,i},2));
    F{1,i} = zeros(m,size(S{1,i},2));
    Y1{1,i} = zeros(Dim,size(S{1,i},2));
    Y2{1,i} = zeros(Dim,d);
    Y4{1,i} = zeros(m,size(S{1,i},2));
    Y1_tmp{1,i} =zeros(Dim,size(S{1,i},2));
    Y2_tmp{1,i} = zeros(Dim,d);
    Y4_tmp{1,i} = zeros(m,size(S{1,i},2));
end
while iter < Maxiter
    iter
    iter = iter+1;
    for i=1:M
        temp = Z{1,i} + Y4{1,i}/mu;
        gpu_tmp = gpuArray(temp);
        [U_Temp,sigma_Temp,V_Temp] = svd(gpu_tmp,'econ');
        gpu_U_Temp = gather(U_Temp);
        gpu_sigma = gather(sigma_Temp);
        gpu_V_Temp = gather(V_Temp);
        sigma = diag(gpu_sigma);
        svp = length(find(sigma>1/mu));
        if svp>=1
            sigma = sigma(1:svp)-1/mu;
        else
            svp = 1;
            sigma = 0;
        end
        F{1,i} = gpu_U_Temp(:,1:svp)*diag(sigma)*gpu_V_Temp(:,1:svp)';
    end
    for i=1:M
        Wpt1 = S{1,i}*S{1,i}'+eye(d);
        Wpt2 = (W*T*Z{1,i}+Ez{1,i})*S{1,i}'+W+Ew{1,i}-(Y1{1,i}*S{1,i}'+Y2{1,i})/mu;
        temp = Wpt2/ Wpt1;
        Wi{1,i} = temp;
    end
    for i=1:M
        Zpt1 = T'*(W'*W)*T+eye(m);
        Zpt2 = (T'*W'*Y1{1,i}-Y4{1,i})/mu + F{1,i}+T'*W'*(Wi{1,i}*S{1,i}-Ez{1,i});
        temp = Zpt1\Zpt2;
        Z{1,i} = temp;
    end
    temp = W + Y3/mu;
    gpu_tmp = gpuArray(temp);
    [U_Temp,sigma_Temp,V_Temp] = svd(gpu_tmp,'econ');
    gpu_U_Temp = gather(U_Temp);
    gpu_sigma = gather(sigma_Temp);
    gpu_V_Temp = gather(V_Temp);
    sigma = diag(gpu_sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = gpu_U_Temp(:,1:svp)*diag(sigma)*gpu_V_Temp(:,1:svp)';
    for i=1:M
        temp = Wi{1,i}*S{1,i}-W*T*Z{1,i}+ Y1{1,i}/mu;
        Ez{1,i} = max(0,temp - alaph/mu)+min(0,temp + alaph/mu);
    end
    for i=1:M
        temp = Wi{1,i}-W+Y2{1,i}/mu;
        Ew{1,i} = max(0,temp - beta/mu)+min(0,temp + beta/mu);
    end
    temp = zeros(d);
    for i=1:M
        temp = temp + T*Z{1,i}*Z{1,i}'*T'+eye(d);
    end
    Wpt1 = temp + eye(d);
    temp = zeros(Dim,d);
    for i=1:M
        temp = temp + Y1{1,i}*Z{1,i}'*T'+Y2{1,i};
    end
    Wpt2 = (temp+ mu*J-Y3)/mu;
    temp = zeros(Dim,d);
    for i=1:M
        temp = temp + (Wi{1,i}*S{1,i}-Ez{1,i})*Z{1,i}'*T'+Wi{1,i}-Ew{1,i};
    end
    Wpt2 = Wpt2 + temp;
    W = Wpt2/Wpt1;
end