import tensorflow as tf
import numpy as np
normal_variable=tf.Variable(tf.random_normal([2,3]),name="variable1")
with tf.variable_scope("scope1"):

    first_variable_in_scope1=tf.get_variable("variable1",[1])
    tf.get_variable_scope().reuse_variables()
    second_variable_in_scope1=tf.get_variable("variable1",[1])

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(normal_variable.name)

print(normal_variable.eval())
print(first_variable_in_scope1.name)
print(first_variable_in_scope1.eval())
print(second_variable_in_scope1.name)
print(second_variable_in_scope1.eval())

test_mat=np.matrix("1,2,3;4 5 6")
print(np.shape(test_mat))
print(test_mat)
test_var=tf.Variable(test_mat)
sess.run(tf.global_variables_initializer())
print(test_var.eval())
test_ten=tf.get_variable(name="test_ten",initializer=tf.random_normal_initializer(),shape=[2,3])
sess.run(tf.global_variables_initializer())
print(test_ten.eval())
print(test_ten.get_shape())
