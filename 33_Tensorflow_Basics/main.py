import tensorflow as tf
tf.compat.v1.disable_eager_execution() 

# creates nodes in a graph
# "construction phase"
x1 = tf.constant(5) 
x2 = tf.constant(6)

# result = x1 * x2
result = tf.multiply(x1, x2)

# Without the "with" statement
# sess = tf.compat.v1.Session()
# print(sess.run(result))
# sess.close()

# defines our session and launches graph
with tf.compat.v1.Session() as sess:
    output = sess.run(result)
    # print(sess.run(result))
    print(output)

print(output)