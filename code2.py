import cv2
import numpy as np
import glob

images = glob.glob("left\*.jpg")

criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
obj_points = []    # 存储3D点
img_points = []    # 存储2D点
# 获取标定板角点的位置
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
# print(objp)
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)  # 在原角点的基础上寻找亚像素角点
        img_points.append(corners2)
        cv2.drawChessboardCorners(img, (7,6), corners, ret)   # OpenCV的绘制函数一般无返回值
        cv2.imshow('img', img)
        cv2.waitKey(500)

print (len(img_points))

cv2.destroyAllWindows()
a = np.array(obj_points)
b = np.array(img_points)
b = np.squeeze(b)
[x,y,z] = b.shape
test = np.ones(shape=[x,y,z+1])
test[:,:,:2] = b
test[:,:,2] = 1
b = test

a[:,:,2] = 1
H_img = []
V_img = []

#获取相应矩阵进行标定
for i in range(len(img_points)):
    H = cv2.findHomography(a[i,:,:],b[i])
    H = np.array(H[0])
    H_img.append(H)
    #求 v11 v12 v22
    i,j = 0,0
    v11 = [H[i, 0] * H[j, 0], H[i, 0] * H[j, 1] + H[i, 1] * H[j, 0], H[i, 1] * H[j, 1],
           H[i, 2] * H[j, 0] + H[i, 0] * H[j, 2], H[i, 2] * H[j, 1] + H[i, 1] * H[j, 2], H[i, 2] * H[j, 2]]
    i,j = 0,1
    v12 = [H[i, 0] * H[j, 0], H[i, 0] * H[j, 1] + H[i, 1] * H[j, 0], H[i, 1] * H[j, 1],
           H[i, 2] * H[j, 0] + H[i, 0] * H[j, 2], H[i, 2] * H[j, 1] + H[i, 1] * H[j, 2], H[i, 2] * H[j, 2]]
    i,j = 1,1
    v22 = [H[i, 0] * H[j, 0], H[i, 0] * H[j, 1] + H[i, 1] * H[j, 0], H[i, 1] * H[j, 1],
           H[i, 2] * H[j, 0] + H[i, 0] * H[j, 2], H[i, 2] * H[j, 1] + H[i, 1] * H[j, 2], H[i, 2] * H[j, 2]]
    v11 = np.array(v11)
    v12 = np.array(v12)
    v22 = np.array(v22)

    V_img.append(v12)
    V_img.append(np.subtract(v11,v22))

V_img = np.array(V_img)
k = np.dot(V_img.T,V_img)#将Vb=0 转化为 kb=V'Vb=0,这样k就是一个对阵正定阵
U,sigma,V = np.linalg.svd(k)
b = V[:,5]

#构造矩阵B
B = np.array([ [b[0],b[1],b[3]],[b[1],b[2],b[4]],[b[3],b[4],b[5]] ])
tmp = np.linalg.eigvalsh(B)
#由原论文附录B可得内参数矩阵的构造
v0 = ( B[0,1]*B[0,2] - B[0,0]*B[1,2] )/ (B[0,0]*B[1,1]-B[1,2]*B[1,2])
lamda = B[2,2] - (B[0,2]*B[0,2] + v0*(B[0,1]*B[0,2] - B[0,0]*B[1,2] ) )/B[0,0]
alpha = np.sqrt(lamda / B[0,0])
belta = np.sqrt( lamda * B[0,0] / (B[0,0]*B[1,1] - B[0,1]*B[0,1]) )
c = -B[0,1] * alpha * alpha * belta / lamda
u0 = c * v0 / alpha - B[0,2] * alpha * alpha / lamda

A = np.array([ [alpha , c , u0], [0,belta,v0],[0,0,1]])

#计算外参数
D = []
d = []
Rm = []
Tm = []
H_arr = np.array(H_img)
for i in range(len(img_points)):
    s = (1 / np.linalg.norm( np.linalg.inv(A)* H_arr[i,:,1].T ) + 1/ np.linalg.norm(np.linalg.inv(A)*H_arr[i,:,2]) ) / 2
    r1 = s * np.linalg.inv(A) * H_arr[i,:,1]
    r2 = s * np.linalg.inv(A) * H_arr[i,:,2]
    r3 = np.cross(r1,r2)
    t = s * np.linalg.inv(A) * H_arr[i,:,2]
    #求出了旋转矩阵
    RL = [r1,r2,r3]
    #优化旋转矩阵
    [U,S,V] = np.linalg.svd(RL)
    RL = U*V.T
    RT = [r1,r2,t]
    XY = RT * a[i].T #M为世界坐标，XY为相机坐标
    UV = A * XY #UV即为最终的像素坐标
    UV[1, :] = UV[1, :] / UV[3, :]
    UV[2, :] = UV[2, :] / UV[3, :]
    UV[3, :] = UV[3, :] / UV[3, :]
    XY[1, :] = XY[1, :] / XY[3, :]
    XY[2, :] = XY[2, :] / XY[3, :]
    XY[3, :] = XY[3, :] / XY[3, :]
    #以下子循环确定求畸变系数方程组的系数矩阵D和常数阵d，依据见Zhang论文 P7
    # Dk = d
    for j in range(len(a[i].shape[1])):
        D11 = (UV[0,j] - u0 )*( np.power( XY(0,j),2 ) + np.power( XY(1,j),2))
        D12 = (UV[0, j] - u0) * np.power((np.power(XY(0, j), 2) + np.power(XY(1, j), 2)),2)
        D21 = (UV[1, j] - v0) * (np.power(XY(0, j), 2) + np.power(XY(1, j), 2))
        D22 = (UV[1, j] - v0) * (np.power(XY(0, j), 2) + np.power(XY(1, j), 2))
        d1 = img_points[i,j,0,0] - UV[0,j]
        d2 = img_points[i, j, 0, 1] - UV[1, j]
        D.append([D11,D12],[D21,D22])
        d.append(d1,d2)

    #以下定义的字符（至Rm），在最大似然估计中用到
    r13 = RL[0,2]
    r12 = RL[0,1]
    r23 = RL[1,2]
    Q1 = - np.arcsin(r13)
    Q2 = np.arcsin(r12/np.cos(Q1))
    Q3 = np.arcsin(r23/np.cos(Q1))

    #使用tensorflow求解
