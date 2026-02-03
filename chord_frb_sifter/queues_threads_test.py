from threading import Thread
from queue import SimpleQueue, Empty
import time
import requests

'''

This is a little experimental script for an Action Picker type
situation where we want to perform a handful of different operations
that could potentially block, so we want to put requests into queues
and have them be handled in sequence by different worker threads.

At first I was thinking about just having a set of Queues with worker
Threads that would pull work from the queues -- eg, making an HTTP
request, or doing a database update.  But if you want to get a
success/failure message back, you need to use a second set of queues
for the return values, figure out some unique IDs for events, and
pretty soon you have reinvented an async processing library...

... like concurrent.futures!

https://docs.python.org/3/library/concurrent.futures.html

'''

import concurrent.futures as cf
import numpy as np
import threading

# Not strictly necessary, but the requests.Session object lets us do
# still like keep Cookie state, or re-use TCP connections for
# efficiency.
session = requests.Session()

# Here's an example database update.
def update_db(args):
    print('Starting db update:', args, 'in thread', threading.current_thread().name)
    time.sleep(1)
    # maybe it fails?
    if np.random.uniform() < 0.25:
        print('DB update failed for', args)
        raise RuntimeError("Databases, man, you just can't count on them")
    print('Updated db:', args)

# Here's an example callback using REST
def fetch_http(args):
    print('Making http request for', args, 'in thread', threading.current_thread().name)
    r = session.get('http://httpbin.org/get?event=%i' % args)
    time.sleep(1)
    # maybe it fails?
    u = np.random.uniform()
    if u < 0.25:
        print('Http request failed for', args)
        raise RuntimeError('HTTP request failed')
    print('Http request finished for', args)
    return r

# We'll call this function (eg, in the actor code) to queue an
# intensity-data callback.  This will call the "fetch_http" method
# in a worker thread with the given args.
def do_http_callback(exe, args):
    '''
    exe: async executor
    '''
    exe.submit(fetch_http, args)

# Called from the actor code to fire off a database update,
# by calling update_db with the given args, in a worker thread.
def do_db_update(exe, args):
    '''
    exe: async executor
    '''
    exe.submit(update_db, args)

def main():
    executor = cf.ThreadPoolExecutor(max_workers=3)

    for i in range(1, 11):
        do_http_callback(executor, (i,))
    for i in range(1, 5):
        do_db_update(executor, (i,))
    time.sleep(5)

    print('Shutdown starting')
    executor.shutdown(wait=True)
    print('Shutdown finished')

if __name__ == '__main__':
    main()
