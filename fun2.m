function result = fun2(para,m,M)
    for i=1:11
        data = para((i-1)*6+1:i*6);
        R(i,:) = data(1:3);
        T(i,:) = data(4:6);
    end
%     k(1) = para(11*6+1);
%     k(2) = para(11*6+2);
    k = [0,0];
    alpha_u = 0;
    skewness = 0;
    alpha_v = 0;
    u0 = 0;
    v0 = 0;
    
    k(1),k(2),alpha_u,skewness,u0,alpha_v,v0 = para(11*6+1:73);
    result = [];
    for i=1:11
        for j =1:42
            data = M(:,j);
            TMP1 = data.*R(i,:)'+T(i,:)';
            TMP2 = TMP1/TMP1(3);
            r2 = TMP2(1)*TMP2(1) + TMP2(2)*TMP2(2);
            x2 = TMP2(1) * (1 + k(1)*r2 + k(2)*r2*r2 );
            y2 = TMP2(2) * (1 + k(1)*r2 + k(2)*r2*r2 );
            u = alpha_u * x2 + u0;
            v = alpha_v * y2 + v0;
            tmp = (m(1,j,i)-u);
            out = (m(1,j,i)-u).^2 + (m(2,j,i)-v).^2;
            result = [result, sum(out)];
        end
    end
end