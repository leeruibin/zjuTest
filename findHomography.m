function H = findHomography(M,m)
%获取数据点数    
NP = size(M,2);
A = ones(2*NP,9);

for i=1:NP
    A(2*i-1,:) = [M(:,i)', 0,0,0, -m(1,i)*M(:,i)' ];
    A(2*i,:)   = [0,0,0,M(:,i)', -m(2,i)*M(:,i)' ];
end
[S,V,D] = svd(A);
H = D(:,9);
H = H / H(9);

options = optimoptions('lsqnonlin', 'Algorithm','levenberg-marquardt');% 使用lsqnonlin进行非线性最小二乘求解
x = lsqnonlin( @fun1, reshape(H,1,9) , [],[],options, m, M);
H=reshape(x,3,3);         % 将矩阵H恢复为3*3
H=H';
end

