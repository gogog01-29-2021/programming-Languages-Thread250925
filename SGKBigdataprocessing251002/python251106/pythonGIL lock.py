"""
Problem Formulation & set up 
Example: Crunching Primes

Sum all of the prime numbers in range of integers starting from 1,000,000 to 5,000,000

Experimental setup
8 Core Mac Pro
8 GB ram
Python 2.6

1.Multi Threading
2.Multi processing Library
"""





//Single Threading
import math

def isprime(n):
    # Returns True if n is prime and false otherwise
    if not isinstance(n, int):
        raise TypeError('not int')
    if n < 2:
        return False
    if n == 2:
        return True
    max = int(math.ceil(math.sqrt(n)))
    i = 2
    while i <= max:
        if n % i == 0:
            return False
        i += 1
    return True

def sum_primes(n):
    # Calculates sum of all primes below given integer n
    # Range start from 2 to n, check is prime
    return sum([x for x in xrange(2, n) if isprime(x)])

for i in xrange(100000, 5000000, 100000):
    print sum_primes(i)



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
