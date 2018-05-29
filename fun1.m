function result = fun1(H,m,M)
    T=reshape(H,3,3);         % ½«¾ØÕóH»Ö¸´Îª3*3
    T=T';
    MM = T*M;
    result = sum(MM./(MM(3,:)) - m,2);
end