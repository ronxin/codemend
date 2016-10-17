"""
http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing
"""
import time
from multiprocessing import Process, RawValue as Value, Lock

class Counter(object):
    def __init__(self, initval=0, name=None):
        self.val = Value('i', initval)
        self.lock = Lock()
        self.name = name

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value

    def incrementGet(self):
        with self.lock:
            self.val.value += 1
            return self.val.value

    def __repr__(self):
        if self.name:
            return self.name + ': ' + str(self.value())
        else:
            return str(self.value())

def func(counter):
    for i in range(50):
        time.sleep(0.01)
        counter.increment()

if __name__ == '__main__':
    counter = Counter(0)
    procs = [Process(target=func, args=(counter,)) for i in range(10)]

    for p in procs: p.start()
    for p in procs: p.join()

    print counter.value()
