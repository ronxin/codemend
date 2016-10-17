"""
http://stackoverflow.com/questions/31255118/python-process-pool-with-timeout-on-each-process-not-all-of-the-pool
"""
import multiprocessing as mp

def run_with_timeout(timeout, func, *args):
    receive_end, send_end = mp.Pipe(duplex=False)
    p = mp.Process(target=func, args=args, kwargs=dict(send_end=send_end))
    p.daemon = True
    p.start()
    send_end.close()  # child must be the only one with it opened
    p.join(timeout)
    if p.is_alive():
        # debug('%s timeout', args)
        p.terminate()
    else:
        return receive_end.recv()  # get value from the child

import logging
import time
from functools import partial
from multiprocessing.pool import ThreadPool

debug = logging.getLogger(__name__).debug

def run_mp(n, send_end):
    start = time.time()
    debug('%d starting', n)
    try:
        time.sleep(n)
    except Exception as e:
        debug('%d error %s', n, e)
    finally:
        debug('%d done, elapsed: %.3f', n, time.time() - start)
    send_end.send({n: n*n})

if __name__=="__main__":
    tasks = [1, 2, 4, 1, 8, 2]

    logging.basicConfig(format="%(relativeCreated)04d %(message)s", level=logging.DEBUG)
    print(ThreadPool(processes=4).map(partial(run_with_timeout, 3, run_mp), tasks))
