#import tensorflow
import tensorflow as tf
import time
#empty list of data files
filenames=[]
#add datafiles to the list
filenames.append("dataset/test1.bin")
filenames.append("dataset/test2.bin")
filenames.append("dataset/test3.bin")
filenames.append("dataset/test4.bin")
#create queue of filenames
filename_queue=tf.train.string_input_producer(filenames,shuffle=False,num_epochs=2,capacity=2)
#initialize a reader
reader=tf.FixedLengthRecordReader(3073,name="reader")
#read records from the filenames stored in filename_queue
key,value=reader.read(filename_queue,name="value")
#decode the read records to uint8 format
bytes=tf.decode_raw(value,tf.uint8)
#Slice the label(first byte of record) off the read and decoded records
label = tf.cast(tf.slice(bytes, [0], [1]), tf.float32,name="label")
#Reshape to desired shape
#depth_major = tf.reshape(tf.slice(bytes, [1],[4]),[1, 2, 2])
#uint8image = tf.transpose(depth_major, [1, 2, 0])

#image = tf.cast(uint8image, tf.float32)
uint8image=tf.slice(bytes,[1],[4])
image = tf.cast(uint8image
                , tf.float32)
#batching data
images, label_batch = tf.train.batch([image, label],batch_size=2,num_threads=1,capacity=1000,name="batching")

labels=tf.reshape(label_batch,[2])




#session object
sess=tf.Session()
#initializing variables
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
#intializing coordinators
coord=tf.train.Coordinator()
#starting queue runners
threads=tf.train.start_queue_runners(sess=sess)
i=0
total_time=0
#Running training step
try:
    while not coord.should_stop():
        start_time=time.time()
        print("running",i)

        print(sess.run(images))
        i=i+1
        duration = time.time() - start_time
        total_time=total_time+duration
        print(duration)
except tf.errors.OutOfRangeError:
    print("limit")
    print("Average duration:", total_time / i)

finally:
    coord.request_stop()
coord.join(threads)
sess.close()
