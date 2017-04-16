# import tensorflow
import tensorflow as tf
# total bytes per image
TOTAL_BYTES = 3073
# create a list of filenames
filenames = ["dataset/data_batch_%d.bin" % i for i in range(1, 6)]
# create a queue of filenames
filename_queue = tf.train.string_input_producer(filenames,shuffle=False)
# initialize a reader to read TOTAL_BYTES bytes
reader = tf.FixedLengthRecordReader(TOTAL_BYTES)
# read TOTAL_BYTES bytes from the files
key, value = reader.read(filename_queue)
# decode read bytes to perceivable datatype
vector_bytes = tf.decode_raw(value, tf.uint8)
#create a session object
sess=tf.InteractiveSession()
#start queue runners
tf.train.start_queue_runners()
#print uint8 value of data read
print("vector_bytes ",sess.run(vector_bytes))
#print the number of elements in list vector_bytes
print("number of elements in vector_bytes ",len(sess.run(vector_bytes)))
label = tf.cast(tf.slice(vector_bytes, [0], [1]), tf.int32)



depth_major = tf.reshape(
    tf.slice(vector_bytes, [1],
                     [3072]),
    [3, 32, 32])
uint8image = tf.transpose(depth_major, [1, 2, 0])

#print(sess.run(uint8image))
A=tf.constant([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
print(A.get_shape())

def _as_tensor_list(tensors):
    if isinstance(tensors,dict):
        print(1)
        return [tensors[k] for k in sorted(tensors)]
    else:
        return tensors
print(sess.run(_as_tensor_list(A)))