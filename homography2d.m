
% 
% *********************************************************************************** 
% *******          A Flexible New Technique for Camera Calibration            ******* 
% *********************************************************************************** 
%                            7/2004    Simon Wan 
%                            //2006-03-04 �������ʣ�simonwan1980@gmail.com (��Ϊ�Ѵӹ������ҵ���˵�ַ������simonwan1980@hit.edu.cn) 
% 
% REF:	   "A Flexible New Technique for Camera Calibration" 
%           - Zhengyou Zhang  
%           - Microsoft Research  
% 
% HOMOGRAPHY2D - computes 2D homography 
% 
% Usage:   H = homography2d(x1, x2) 
%          H = homography2d(x) 
% 
% Arguments: 
%          x1  - 3xN set of homogeneous points 
%          x2  - 3xN set of homogeneous points such that x1<->x2 
%          
%           x  - If a single argument is supplied it is assumed that it 
%                is in the form x = [x1; x2] 
% Returns: 
%          H - the 3x3 homography such that x2 = H*x1 
% 
% This code follows the normalised direct linear transformation  
% algorithm given by Hartley and Zisserman "Multiple View Geometry in 
% Computer Vision" p92. 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% ���ӳ�����Դ���£� 
% Peter Kovesi 
% School of Computer Science & Software Engineering 
% The University of Western Australia 
% pk at csse uwa edu au 
% http://www.csse.uwa.edu.au/~pk 
% 
% May 2003  - Original version. 
% Feb 2004  - single argument allowed for to enable use with ransac. 
%H=A*[r1,r2,t]; 
function H = homography2d(varargin) 
     
    [x1, x2] = checkargs(varargin(:));% varargin"�䳤�����������б�"varargin�����Ǹ�Ԫ������ 
    M=x1;                             % varargout"�䳤����������б�" 
    m=x2; 
    % Attempt to normalise�� ��񻯣�each set of points so that the origin  
    % is at centroid �����ģ�and mean distance from origin is sqrt(2).����Ϊ�������Σ� 
    [x1, T1] = normalise2dpts(x1); 
    [x2, T2] = normalise2dpts(x2); 
     
    % Note that it may have not been possible to normalise 
    % the points if one was at infinity so the following does not 
    % assume that scale parameter w = 1. 
    % Estimation of the H between the model plane and its image, P18������Ӧ�Ծ��� 
    Npts = length(x1); 
    A = zeros(3*Npts,9);%AΪ�������� 
     
    O = [0 0 0]; 
    for n = 1:Npts 
	X = x1(:,n)';%����  
	x = x2(1,n);y = x2(2,n); w = x2(3,n); 
	A(3*n-2,:) = [  O  -w*X  y*X]; 
	A(3*n-1,:) = [ w*X   O  -x*X]; 
	A(3*n  ,:) = [-y*X  x*X   O ]; 
    end 
     
    [U,D,V] = svd(A); 
    % Ax=b  x=A\b; 
    % Extract homography��Ӧ�Ծ��� 
    H1 = reshape(V(:,9),3,3)' 
            
    % Denormalize������, 
    H2= T2\H1*T1; 
    H=H2/H2(3,3); 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % Maximun likelihood estimation for the H�����Ȼ���� 
    % using the function(10), P7 
    options = optimset('LargeScale','off','LevenbergMarquardt','on'); 
    [x,resnorm,residual,exitflag,output]  = lsqnonlin( @simon_H, reshape(H,1,9) , [],[],options,m, M); 
    H=reshape(x,3,3); 
    H=H/H(3,3);    
 
Row = size(M,2); 
Temp = zeros(Row*2,8); 
Temp(1:Row,1) = M(1,:); 
Temp(1:Row,2) = M(2,:); 
Temp(1:Row,3) = M(3,:); 
Temp(Row+1:Row*2,4) = M(1,:); 
Temp(Row+1:Row*2,5) = M(2,:); 
Temp(Row+1:Row*2,6) = M(3,:); 
Goal = zeros(Row,1); 
for i=1:Row 
    Temp(i,7) = -m(1,i,1)*M(1,i); 
    Temp(i,8) = -m(1,i,1)*M(2,i); 
     
    Temp(Row+i,7) = -m(2,i,1)*M(1,i); 
    Temp(Row+i,8) = -m(2,i,1)*M(2,i); 
    Goal(i) = -m(1,i,1)*M(3,i); 
    Goal(Row+i) = -m(2,i,1)*M(3,i); 
end 
 
HH = [];HH = inv(Temp'*Temp)*Temp'*Goal; 
kk = [];kk = Temp*HH-Goal;sumkk = kk'*kk; 
HH1 = [];HH1 = [H(1,:),H(2,:),H(3,:)]'; 
kk1 = [];kk1 = -Temp*HH1(1:8) - Goal;sumkk1 = kk1'*kk1; 
[sumkk sumkk1] 
 
 
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%-------------------------------------------------------------------------- 
% Function to check argument values and set defaults 
 
function [x1, x2] = checkargs(arg); 
     
    if length(arg) == 2 
	x1 = arg{1}; 
	x2 = arg{2}; 
	if ~all(size(x1)==size(x2)) 
	    error('x1 and x2 must have the same size'); 
	elseif size(x1,1) ~= 3 
	    error('x1 and x2 must be 3xN'); 
	end 
	 
    elseif length(arg) == 1 
	if size(arg{1},1) ~= 6 
	    error('Single argument x must be 6xN'); 
	else 
	    x1 = arg{1}(1:3,:); 
	    x2 = arg{1}(4:6,:); 
	end 
    else 
	error('Wrong number of arguments supplied'); 
    end 