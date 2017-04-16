import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
IMAGE_BYTES=3072
LABEL_BYTES=1
filenames=["dataset/data_batch_%d.bin"%i for i in range(1,6)]
contents=tf.read_file("dataset/data_batch_1.bin")
filename_queue=tf.train.string_input_producer(filenames)
TOTAL_BYTES=IMAGE_BYTES+LABEL_BYTES
reader=tf.FixedLengthRecordReader(record_bytes=TOTAL_BYTES)
key,value=reader.read(filename_queue)
vector_bytes=tf.decode_raw(value,tf.uint8)
sess=tf.InteractiveSession()
tf.train.start_queue_runners()
#print(sess.run([filename_queue.dequeue()]*2))
#print(sess.run([filename_queue.dequeue(),filename_queue.dequeue()]))
sess.run([contents])


#print(vector_bytes.eval())
#print(vector_bytes)
#print("vector_bytes is",vector_bytes.eval())

#A=tf.slice(vector_bytes, [0], [40])
#print(sess.run([vector_bytes]*3))
#print(sess.run([vector_bytes,vector_bytes,vector_bytes]))
#print(sess.run(vector_bytes))





#print(A.eval())
label = tf.cast(tf.slice(vector_bytes, [0], [1]), tf.int32)
print(sess.run([value,vector_bytes,label]))

#print("label is",label.eval())
#print(label.eval())
depth_major = tf.reshape(
    tf.slice(vector_bytes, [LABEL_BYTES],
                     [IMAGE_BYTES]),
    [3, 32, 32])
uint8image = tf.transpose(depth_major, [1, 2, 0])
print("depth_major is",depth_major.eval())
print("uint8image_is",uint8image.eval())
print("shape of uint8image_is",uint8image.get_shape())
reshaped_image = tf.cast(uint8image, tf.float32)
print("reshaped is",reshaped_image.eval())
resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         24, 24)
print("ye")
print(resized_image.eval())
float_image = tf.image.per_image_standardization(resized_image)
print(float_image.eval())
print(float_image.get_shape())

print("le")

float_image.set_shape([24, 24, 3])
print(float_image.get_shape())

print(float_image.eval())



