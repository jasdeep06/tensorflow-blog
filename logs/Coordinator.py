#import tensorflow
import tensorflow as tf
#import threading
import threading
import time
#add a queue to the graph
q=tf.FIFOQueue(capacity=40000,dtypes=tf.int32,name="queue")
#generate a random number
random_number=tf.random_uniform(shape=(),dtype=tf.int32,maxval=50)
#add enqueue operation to graph
enqueue_op=q.enqueue(random_number,"enqueue_op")
size_op=q.size()
dequeue_op=q.dequeue()
#initialize a session
sess=tf.Session()
#define enqueue_random function to run enqueue op

def enqueue_dequeue(coord):
    for i in range(20):
        if coord.should_stop():
            break
#run enqueue_op
        print([sess.run(enqueue_op),sess.run(size_op)])
    for i in range(40):
        if coord.should_stop():
            break
        #print(threading.current_thread().name," Starting to dequeue")
        print([sess.run(dequeue_op),sess.run(size_op)])
        if sess.run(size_op)<1:
            #sess.run(q.close())
            #coord.request_stop()
            print("Requested to stop")

coord=tf.train.Coordinator()

#create 2 threads
threads=[threading.Thread(target=enqueue_dequeue,args=(coord,)) for i in range(2)]
#note the time of starting of threads
start=time.time()
#start the threads
[t.start() for t in threads]
coord.join(threads)
print(time.time()-start)