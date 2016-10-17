"""
A template for the baseline methods that participate in the evaluation of
eval2.py.

"""

from abc import ABCMeta, abstractmethod
import random
from recordclass import recordclass
import re

SuggestItem = recordclass('SuggestItem', ['elem', 'score'])

class Context:
  """
  Vehicle of all the necessary information to be included in a query for the
  model to make a prediction.

  This abstract representation is composed only of functions called, parameter
  used, return relations, etc.

  There are a corresponding preprocessor and postprocessor (e.g., matching
  model's suggested highlight with user's cursor position, etc.)

  This class itself should not contain any thing about the messy world.

  """
  def __init__(self, used_elem_ids, used_elem_objs=None, var_map=None):
    """
    SIMPLIFIED elements (string IDs) only.

    """
    self._used_elem_ids = used_elem_ids
    self._used_elem_objs = used_elem_objs
    self._var_map = var_map

  def used_elems(self):
    """
    A list of strings. Simplified element IDs.

    """
    return self._used_elem_ids

  def used_elem_objs(self):
    return self._used_elem_objs

  def var_map(self):
    return self.var_map

class Baseline(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def suggest(self, query, context):
    """
    Returns an ordered list of SuggestItem.

    """
    pass

  @abstractmethod
  def __repr__(self):
    pass

class RandomBaseline(Baseline):
  def __init__(self, all_elems, mode='all'):
    """
    Parameters
    ----------
    - all_elems: a list of elements (string)
    - mode: 'all' | 'used_only' | 'used_extend'

    """
    self.all_elems = list(all_elems)
    self.mode = mode

  def suggest(self, query, context):
    mode = self.mode

    if mode == 'all':
      candidates = self.all_elems

    elif mode == 'used_only':
      candidates = context.used_elems

    elif mode == 'used_extend':
      r = re.compile('^' + '|'.join(re.escape(p) for p in context.used_elems))
      candidates = filter(lambda x: r.match(x), self.all_elems)

    else:
      raise ValueError('Unrecognized mode: %s'%mode)

    sample = random.sample(candidates, 50)

    return [SuggestItem(elem=x, score=0) for x in sample]

  def __repr__(self):
    return 'RANDOM'


if __name__ == '__main__':
  import string
  rand_b = RandomBaseline(string.letters)
  suggest = rand_b.suggest(None, None)
  for item in suggest:
    print item.elem, item.score

  print 'Should see a bunch a single letters with 0 weights'
