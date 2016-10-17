"""
A template for the baseline methods that participate in the evaluation of
eval.py.

"""

from abc import ABCMeta, abstractmethod

class Baseline(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def rank_funcs(self, query, funcs, parent):
    """
    Returns an ordered list of tuples. The first element of each tuple should
    be func.
    """
    pass

  @abstractmethod
  def rank_args(self, query, func, args, parent):
    """
    Returns an ordered list of tuples. The first element of each tuple should
    be arg.
    """
    pass

  @abstractmethod
  def __repr__(self):
    pass

import random
class RandomBaseline(Baseline):

  def rank_funcs(self, query, funcs, parent):
    funcs = list(funcs)
    random.shuffle(funcs)
    return map(lambda x:(x,), funcs)  # convert to a list of tuples

  def rank_args(self, query, func, args, parent):
    args = list(args)
    random.shuffle(args)
    return map(lambda x:(x,), args)

  def __repr__(self):
    return 'RANDOM'
