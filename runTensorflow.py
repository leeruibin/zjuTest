import numpy as np
import tensorflow as tf

#尝试训练 原始数据
one = np.float32(np.ones(1,42*11))
M = np.float32(np.random.rand(4,42*11))
m = np.float32(np.random.rand(3,42*11))

RT = tf.Variable(tf.zeros([3,4]))
A = tf.Variable(tf.zeros([3,3]))
k1 = tf.Variable(tf.zeros([1]))
k2 = tf.Variable(tf.zeros([1]))

r1 = tf.subtract(tf.reduce_sum(tf.square(tf.matmul(RT,M)),axis=2),one)
r2 = tf.square(r1)

K = tf.Variable(np.array([1+k1*r1+k2*r2,1+k1*r1+k2*r2,1]))
tmp = tf.matmul(K,r1)
mm = tf.matmul(A,tmp)
loss = tf.reduce_norm(m-mm)
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

x_data = np.float32(np.random.rand(2,100))
y_data = np.dot([0.100,0.200],x_data) + 0.300

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,2],0,1))
y = tf.matmul(W,x_data) + b
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 ==0:
        print(step, sess.run(W),sess.run(b))