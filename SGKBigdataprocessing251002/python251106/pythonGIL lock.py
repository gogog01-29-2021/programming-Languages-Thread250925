//Multi Threading
from threading import Thread
from Queue import Queue, Empty

def do_work(q):
    while True:
        try:
            x = q.get(block=False)
            print sum_primes(x)
        except Empty:
            break

#main function
work_queue = Queue()
for i in xrange(100000, 5000000, 100000):
    work_queue.put(i)

threads = [Thread(target=do_work, args=(work_queue, )) for i in
           range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()












//Multi processing
from multiprocessing import Process, Queue
from Queue import Empty

work_queue = Queue()
for i in xrange(100000, 5000000, 100000):
    work_queue.put(i)

processes = [Process(target=do_work, args=(work_queue, )) for i in range(8)]
for p in processes:
    p.start()
for p in processes:
    p.join()
