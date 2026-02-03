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

# Not strictly necessary, but the requests.Session object lets us do
# still like keep Cookie state, or re-use TCP connections for
# efficiency.
session = requests.Session()

# Here's an example database update.
def update_db(args):
    print('Starting db update:', args)
    time.sleep(1)
    # maybe it fails?
    if np.random.uniform() < 0.25:
        raise RuntimeError("Databases, man, you just can't count on them")
    print('Updated db:', args)

# Here's an example callback using REST
def fetch_http(args):
    print('Making http request for', args)
    r = session.get('http://httpbin.org/get?event=%i' % args)
    time.sleep(1)
    # maybe it fails?
    u = np.random.uniform()
    if u < 0.25:
        raise RuntimeError('HTTP request failed')
    print('Http request finished for', args)
    return r

# We'll call this function (eg, in the actor code) to queue an
# intensity-data callback.
def do_http_callback(q, args, on_success, on_failure):
    '''
      q: our work queue
      on_success: function to call upon successful completion:
         gets passed (args, result)
      on_failure: function to call upon failure (exception)
         gets passed (args, exception)

    '''
    q.put((fetch_http, args, on_success, on_failure))

# This is a little wrapper function needed below...
def read_queue(q):
    return q.get()

# The event-managing / queue-handling loop
def manage_events(q, executor):
    # The requests that we're waiting on...
    futures = {}
    q_future = None
    quit_now = False
    while True:
        # We want to wait on either new work arriving on the queue, or
        # a job completing.
        # "q_future" represents new work arriving on the queue.
        if q_future is None:
            q_future = executor.submit(read_queue, q)
        # Now we wait for the event to happen:
        done,not_done = cf.wait([q_future] + list(futures.keys()),
                                timeout=None,
                                return_when=cf.FIRST_COMPLETED)
        # we get back the list of "done" and "not_done" events.

        for f in done:
            # New work arrived on the queue!
            if f is q_future:
                event = f.result()
                print('Got event from queue:', event)
                # reset our queue-waiting object
                q_future = None

                # A special "None" event means "quit now".
                # presumably we would also want a "wrap everything up cleanly and quit"...
                if event is None:
                    # cancel futures?
                    quit_now = True
                    break

                # The event contains:
                target, args, on_success, on_failure = event
                # Submit the work...
                f = executor.submit(target, args)
                # Save the resulting "future"
                futures[f] = (target, args, on_success, on_failure)
                continue

            # A requested bit of work completed!
            target, args, on_success, on_failure = futures[f]
            success = False
            exc = None
            try:
                # if there was an exception, it gets re-thrown now!
                r = f.result()
                success = True
            except Exception as e:
                r = None
                exc = e
            if success:
                if on_success is not None:
                    on_success(args, r)
            else:
                if on_failure is not None:
                    on_failure(args, exc)
            del futures[f]

        if quit_now:
            break
    print('manage_events returning')

# example callbacks
def http_completed(args, result):
    print('callback http_completed:', args, result)

def http_failed(args, exc):
    print('callback http_failed:', args, exc)

def main():
    executor = cf.ThreadPoolExecutor(max_workers=100)

    q = SimpleQueue()
    #t = Thread(target=run_slow_v1, args=(q,), daemon=True)
    t = Thread(target=manage_events, args=(q, executor), daemon=True)
    t.start()

    
    for i in range(1, 11):
        do_http_callback(q, (i,), http_completed, http_failed)
    # q.put(42)
    # time.sleep(3)
    # q.put(52)
    # q.put(62)
    # 
    time.sleep(10)

    print('Shutdown')
    #executor.shutdown(wait=False, cancel_futures=True)
    q.put(None)
    executor.shutdown(wait=True)
    t.join()

if __name__ == '__main__':
    main()
