"""
Plugging bimodal into the eval framework.
"""
import numpy as np

from baseline import Baseline
from bimodal import BiModal
from annotate_code_with_api import findCallNodes, extractCallComponents
from myast import MyAST

class BiModalBaseline(Baseline):
  def __init__(self, model_file):
    self.model = BiModal.load(model_file)
    self.model.random = np.random.RandomState()
    self.name = 'BiModal-%s'%model_file

  def rank_funcs(self, query, funcs, parent):
    """
    Returns an ordered list of tuples. The first element of each tuple should
    be func.
    """
    assert parent.current_node
    call_nodes = findCallNodes(parent.current_node)
    func_scores = {}  # [func] = score
    for call in call_nodes:
      func, keywords = extractCallComponents(call)
      if func in funcs:
        myast_call = MyAST(node=call)
        score = self.model.scoreFullTree(query, myast_call)
        if func not in func_scores or func_scores[func] < score:
          func_scores[func] = score
    sorted_funcs = sorted(func_scores.items(), key=lambda x:x[1], reverse=True)
    return sorted_funcs

  def rank_args(self, query, func, args, parent):
    """
    Returns an ordered list of tuples. The first element of each tuple should
    be arg.
    """
    pass
    # Not implemented
    return map(lambda x:(x,), args)

  def __repr__(self):
    return self.name
