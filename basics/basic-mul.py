import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

x1 = tf.constant([1,2,4,6,7])
x2 = tf.constant([3,6,4,8,1])

result = tf.multiply(x1, x2)

#print(result)

sess = tf.Session()

print(sess.run(result))

sess.close()
# with tf.Session() as sess:
#     output = sess.run(result)
#     print(output)
