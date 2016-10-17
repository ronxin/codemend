"""
Implementation of a baseline method that uses simple whoosh NL index to do
function and argument lookup.

Needs to run ../docstring_parse/index_nl2.py prior to using this.
"""
"""
Handles NL query given function context.

To build the query, use ../docstring_parse/index_nl.py
"""
import sys
import whoosh
import whoosh.index
import whoosh.query

sys.path.append('../docstring_parse')
from consolidate import get_method, get_class

from baseline import Baseline

INDEX_DIR = '../demo/index2'

class NLQueryBaseline(Baseline):
  def __init__(self):
    print 'Loading indexes...'
    self.index_fu = whoosh.index.open_dir(INDEX_DIR, 'fu', True)
    self.index_fau = whoosh.index.open_dir(INDEX_DIR, 'fau', True)
    print 'Indexes loaded.'

    self.searcher_fu = self.index_fu.searcher()
    self.searcher_fau = self.index_fau.searcher()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def close(self):
    """
    Needs to be closed in order to release resources.
    """
    self.searcher_fu.close()
    self.searcher_fau.close()

  def createOrQueryNL(self, query):
    """
    Creates an Or-query for NL
    """
    terms = query.split()
    return whoosh.query.Or([whoosh.query.Variations('utter', term) for term in terms])

  def createOrQueryPL(self, field, candidates):
    """
    Creates an Or-query for PL (funcs or args)
    """
    return whoosh.query.Or([whoosh.query.Term(field, term) for term in candidates])

  def rank_funcs(self, query, funcs, parent=None):
    """
    Will submit a query to Whoosh like this:
      AND (nl_query, pl_query)

      nl_query = OR(term1, term2, ...)
      pl_query = OR(func_id1, func_id2, ...)

    """
    nl_query = self.createOrQueryNL(query)
    pl_query = self.createOrQueryPL('func_id', funcs)
    queryObj = whoosh.query.And([nl_query, pl_query])
    results = self.searcher_fu.search(queryObj, limit=len(funcs))
    scores = [0.] * len(funcs)
    for r in results:
      # TODO: if there are duplicates in funcs, then all scores of the
      # duplicates should be updated.
      idx = funcs.index(r['func_id'])
      scores[idx] = r.score
    sorted_funcs = sorted(zip(funcs, scores), key=lambda x:x[1], reverse=True)
    return sorted_funcs

  def rank_args(self, query, func, args, parent=None):
    """
    Will submit a query to whoosh like this:
      AND (func_id_query, nl_query, pl_query)

      func_id_query = Term(func_id = func)
      nl_query = OR(term1, term2, ...)
      pl_query = OR(arg1, arg2, ...)
    """
    func_id_query = whoosh.query.Term('func_id', func)
    nl_query = self.createOrQueryNL(query)
    pl_query = self.createOrQueryPL('arg', args)
    queryObj = whoosh.query.And([func_id_query, nl_query, pl_query])
    results = self.searcher_fau.search(queryObj, limit=len(args))
    scores = [0.] * len(args)
    for r in results:
      idx = args.index(r['arg'])
      scores[idx] = r.score
    sorted_args = sorted(zip(args, scores), key=lambda x:x[1], reverse=True)
    return sorted_args

  def __repr__(self):
    return 'BM25'

if __name__ == "__main__":
  """Testing NLQueryHandler"""

  with NLQueryBaseline() as nqb:
    results = nqb.rank_funcs('horizontal bar plot', ['bar', 'barh', 'pie'])
    for x in results:
      print x[0], x[1]

    print '-------------'

    results = nqb.rank_args(
      'add shadow to legend',
      'legend',
      ['shadow', 'bbox_to_anchor'])
    for x in results:
      print x[0], x[1]
