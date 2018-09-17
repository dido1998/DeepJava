import tensorflow as tf

a=tf.placeholder(tf.float32,shape=[25,25,3])
c=tf.contrib.layers.conv2d(a,3,5,activation_fn=None);




grad=tf.gradients(c,a);


import numpy as np
a1=np.random.rand(25,25,3)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	g=sess.run(c,feed_dict={a:a1})
	print(g)