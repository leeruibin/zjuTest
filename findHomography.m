function H = findHomography(M,m)
%��ȡ���ݵ���    
NP = size(M,2);
A = ones(2*NP,9);

for i=1:NP
    A(2*i-1,:) = [M(:,i)', 0,0,0, -m(1,i)*M(:,i)' ];
    A(2*i,:)   = [0,0,0,M(:,i)', -m(2,i)*M(:,i)' ];
end
[S,V,D] = svd(A);
H = D(:,9);
H = H / H(9);

options = optimoptions('lsqnonlin', 'Algorithm','levenberg-marquardt');% ʹ��lsqnonlin���з�������С�������
x = lsqnonlin( @fun1, reshape(H,1,9) , [],[],options, m, M);
H=reshape(x,3,3);         % ������H�ָ�Ϊ3*3
H=H';
end

