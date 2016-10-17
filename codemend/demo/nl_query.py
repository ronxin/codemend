import sys
import whoosh
import whoosh.index
import whoosh.query

sys.path.append('../docstring_parse')
from consolidate import get_method, get_class

INDEX_DIR = 'index'

class NLQueryHandler:
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
    self.searcher_fu.close()
    self.searcher_fau.close()

  def findFunction(self, query):
    queryObj = self.createOrQuery(query)
    return self.searcher_fu.search(queryObj)

  def findArg(self, func_id, query):
    orQuery = self.createOrQuery(query)
    funcIdQuery = whoosh.query.Term('func_id', func_id)
    queryObj = whoosh.query.And([orQuery, funcIdQuery])
    return self.searcher_fau.search(queryObj)

  def createOrQuery(self, query):
    terms = query.split()
    return whoosh.query.Or([whoosh.query.Variations('utter', term) for term in terms])

if __name__ == "__main__":
  """Testing NLQueryHandler"""

  with NLQueryHandler() as nqh:
    results = nqh.findFunction('create horizontals bar plot')
    print len(results)
    for r in results[:5]:
      print r.score, r['utter'], r['func_id']

    results = nqh.findArg('matplotlib.axes.Axes.barh', 'add hatching pattern')
    print '-------------'
    print len(results)
    for r in results[:5]:
      print r.score, r['utter'], '-', r['arg']
