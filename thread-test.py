#import tensorflow
import tensorflow as tf
#import threading
import threading
import time
#add a queue to the graph
q=tf.FIFOQueue(capacity=200,dtypes=tf.int32,name="queue")
#generate a random number
random_number=tf.random_uniform(shape=(),dtype=tf.int32,maxval=50)
#add enqueue operation to graph
enqueue_op=q.enqueue(random_number,"enqueue_op")
#initialize a session
sess=tf.Session()
#define enqueue_random function to run enqueue op
def enqueue_random():
    for i in range(20):
#run enqueue_op
	    sess.run(enqueue_op)
#create 2 threads
threads=[threading.Thread(target=enqueue_random,args=()) for i in range(2)]
#note the time of starting of threads
start=time.time()
#start the threads
[t.start() for t in threads]
print(sess.run(q.size()))
time.sleep(0.1)
print(sess.run(q.size()))