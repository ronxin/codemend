"""@deprecated - Use https://github.com/WoLpH/python-progressbar instead."""
import ipywidgets as ipyw
import IPython.display as ipyd
import time

MIN_REPORT_INTERVAL = 1.0  # second
MIN_REPORT_PERCENTAGE = 1.0 / 100

class ProgressBar:
    def __init__(self, iterable, size=None):
        self.it = iter(iterable)
        if not size: size = len(iterable)
        self.size = size
        self.current = 0
        self.next_report_time = time.time() + MIN_REPORT_INTERVAL
        assert type(MIN_REPORT_PERCENTAGE) == float
        self.min_report_interval_count = MIN_REPORT_PERCENTAGE * size
        self.next_report_count = self.min_report_interval_count
        self.f = ipyw.FloatProgress(max=size)
        ipyd.display(self.f)
    
    def __iter__(self):
        return self
    
    def next(self):
        self.current += 1
        if self.current == self.size \
            or self.current <= self.next_report_count \
            or self.next_report_time <= time.time():
                self.report()
        return self.it.next()
    
    def report(self):
        self.f.value = self.current
        self.next_report_time = time.time() + MIN_REPORT_INTERVAL
        self.next_report_count = self.current + self.min_report_interval_count

"""Example (IPython Notebook):
from progressbar import ProgressBar
import time

for i in ProgressBar(xrange(10)):
    time.sleep(0.2)
"""

