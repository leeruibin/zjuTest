clc;
clear;
a = load('data.txt');
b = load('data1.txt');

d2_point = zeros(3,42,11);
for i=1:11
    d2_point(:,:,i) = reshape(b(i,:),[3,42]);
end

d3_point = a';
d3_point = d3_point .* 400;
d3_point(3,:) = 1;
M = d3_point;
num = size(d2_point,3);
npts = size(d3_point,2);
m = d2_point;
for i=1:num
    H(:,:,i) = findHomography(d3_point,d2_point(:,:,i));
end
V=[];
for flag=1:num %每一次取flag代表一幅的vij
    v12(:,:,flag)=[H(1,1,flag)*H(2,1,flag), H(1,1,flag)*H(2,2,flag)+H(1,2,flag)*H(2,1,flag), H(1,2,flag)*H(2,2,flag), H(1,3,flag)*H(2,1,flag)+H(1,1,flag)*H(2,3,flag), H(1,3,flag)*H(2,2,flag)+H(1,2,flag)*H(2,3,flag), H(1,3,flag)*H(2,3,flag)];
    v11(:,:,flag)=[H(1,1,flag)*H(1,1,flag), H(1,1,flag)*H(1,2,flag)+H(1,2,flag)*H(1,1,flag), H(1,2,flag)*H(1,2,flag), H(1,3,flag)*H(1,1,flag)+H(1,1,flag)*H(1,3,flag), H(1,3,flag)*H(1,2,flag)+H(1,2,flag)*H(1,3,flag), H(1,3,flag)*H(1,3,flag)];
    v22(:,:,flag)=[H(2,1,flag)*H(2,1,flag), H(2,1,flag)*H(2,2,flag)+H(2,2,flag)*H(2,1,flag), H(2,2,flag)*H(2,2,flag), H(2,3,flag)*H(2,1,flag)+H(2,1,flag)*H(2,3,flag), H(2,3,flag)*H(2,2,flag)+H(2,2,flag)*H(2,3,flag), H(2,3,flag)*H(2,3,flag)];
    V=[V;v12(:,:,flag);v11(:,:,flag)-v22(:,:,flag)];
end
k=V'*V;          %将Vb=0转化为kb=V'*V*b=0,这样k就是一个对称正定矩阵，可以利用奇异值分解法进行计算
[u,v,d]=svd(V);  %奇异值分解 k=uvd'
b=d(:,6);        %即d的第6列赋予b
v0=(b(2)*b(4)-b(1)*b(5))/(b(1)*b(3)-b(2)^2);
s=b(6)-(b(4)^2+v0*(b(2)*b(4)-b(1)*b(5)))/b(1);
alpha_u=sqrt(s/b(1));
alpha_v=sqrt(s*b(1)/(b(1)*b(3)-b(2)^2));
skewness=-b(2)*alpha_u*alpha_u*alpha_v/s;
u0=skewness*v0/alpha_u-b(4)*alpha_u*alpha_u/s;
A=[alpha_u skewness u0
   0      alpha_v  v0
   0      0        1];

%  [[ 533.55191915    0.          342.41354533]
%  [   0.          533.66184413  232.19354412]
%  [   0.            0.            1.        ]]
% 求 k1 k2 和所有外参, Zhang论文 P6
    D=[];
    d=[];
    Rm=[];
    for flag=1:num
        s=(1/norm(inv(A)*H(1,:,flag)')+1/norm(inv(A)*H(2,:,flag)'))/2;%括号里两项理论上相等，实际不可能，这里取平均数，所以这样求出的R也不会是正交阵
        rl1=s*inv(A)*H(1,:,flag)';
        rl2=s*inv(A)*H(2,:,flag)';
        rl3=cross(rl1,rl2);    %外积
        RL=[rl1,rl2,rl3];      %初步求出了旋转矩阵R，用RL表示
        %%%%%%%%%%%%%%%%%%%%
        %下面2行为优化旋转矩阵（Zhang论文附录C P19）
        [U,S,V] = svd(RL);
        RL=U*V';%主要运用奇异值分解
        %%%%%%%%%%%%%%%%%%%%
        TL=s*inv(A)*H(3,:,flag)';%TL是位移矩阵t
        RT=[rl1,rl2,TL];%即[r1 r2 t]
        %下面是确定（径向）畸变系数
        XY=RT*M;        %M是世界坐标，XY为摄像机坐标
        UV=A*XY;        % sm=A[R t]M,UV就是A[R t]M
        UV=[UV(1,:)./UV(3,:); UV(2,:)./UV(3,:); UV(3,:)./UV(3,:)];%UV第3坐标分量化为1
        XY=[XY(1,:)./XY(3,:); XY(2,:)./XY(3,:); XY(3,:)./XY(3,:)];%XY第3坐标分量化为1，即摄像机坐标
        %以下子循环确定求畸变系数方程组的系数矩阵D和常数阵d，依据见Zhang论文 P7
        for j=1:npts  %前面计算出npts为M列数，就是1幅图的角点数
            D=[D; ((UV(1,j)-u0)*( (XY(1,j))^2 + (XY(2,j))^2 )) , ((UV(1,j)-u0)*( (XY(1,j))^2 + (XY(2,j))^2 )^2) ; ((UV(2,j)-v0)*( (XY(1,j))^2 + (XY(2,j))^2 )) , ((UV(2,j)-v0)*( (XY(1,j))^2 + (XY(2,j))^2 )^2) ];
            d=[d; (m(1,j,flag)-UV(1,j)) ; (m(2,j,flag)-UV(2,j))];
        end
        %以下定义的字符（至Rm），在最大似然估计中用到
        r13=RL(1,3);
        r12=RL(1,2);
        r23=RL(2,3);
        Q1=-asin(r13);
        Q2=asin(r12/cos(Q1));
        Q3=asin(r23/cos(Q1));
        [cos(Q2)*cos(Q1)   sin(Q2)*cos(Q1)   -sin(Q1) ; -sin(Q2)*cos(Q3)+cos(Q2)*sin(Q1)*sin(Q3)    cos(Q2)*cos(Q3)+sin(Q2)*sin(Q1)*sin(Q3)  cos(Q1)*sin(Q3) ; sin(Q2)*sin(Q3)+cos(Q2)*sin(Q1)*cos(Q3)    -cos(Q2)*sin(Q3)+sin(Q2)*sin(Q1)*cos(Q3)  cos(Q1)*cos(Q3)];
        R_new=[Q1,Q2,Q3,TL'];
        Rm=[Rm , R_new];
    end
    
    % Zhang论文 P7（13）
    k=inv(D'*D)*D'*d;%最小二乘解
% 最大似然估计，Zhang论文式(14), P6
    para=[Rm,k(1),k(2),alpha_u,skewness,u0,alpha_v,v0];%包含的信息是R,t,k1,k2,A

options = optimoptions('lsqnonlin', 'Algorithm','levenberg-marquardt');% 使用lsqnonlin进行非线性最小二乘求解

 [x,resnorm,residual,exitflag,output]  = lsqnonlin( @fun2, para, [],[],options, m, M);%第二个参数是初始值；两个空白格是取值的范围区间，可以不写；m，M为已知点
% display the result
    k1=x(num*6+1)
    k2=x(num*6+2)
    A=[x(num*6+3) x(num*6+4) x(num*6+5); 0 x(num*6+6) x(num*6+7); 0,0,1]


% zhang(d3_point,d2_point);


