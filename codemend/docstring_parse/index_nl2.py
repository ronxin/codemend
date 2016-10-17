"""
A variation of index_nl.py.

The differences are:
(1) functions are identified only by its name (i.e., last component in
func_id)
(2) the token of interest (function, or, arg) is added to the NL document to
increase recall.

Creates NL index for F's and FA's using Whoosh.
  - F: functions
  - FA: function-argument

Usage:
    python index_nl2.py
"""
import os
import sys
import whoosh
import whoosh.fields as wf
import whoosh.index

sys.path.append('../models')
from annotate_code_with_api import get_fu_fau

INDEX_DIR = '../demo/index2'

if __name__ == '__main__':
  # Create indices
  if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)

  print 'Getting simplified fu and fau mapping'
  fu, fau = get_fu_fau()

  print 'Creating fu NL index'

  schema_fu = wf.Schema(func_id=wf.ID(stored=True), utter=wf.TEXT(stored=True))
  index_fu = whoosh.index.create_in(INDEX_DIR, schema_fu, 'fu')
  writer_fu = index_fu.writer()
  for f,u in fu.items():
    u += ' ' + f
    f = f.decode('utf-8')
    u = u.decode('utf-8')
    writer_fu.add_document(func_id = f, utter = u)
  writer_fu.commit()

  print 'Creating fau NL index'

  schema_fau = wf.Schema(func_id=wf.ID(stored=True), arg=wf.ID(stored=True), utter=wf.TEXT(stored=True))
  index_fau = whoosh.index.create_in(INDEX_DIR, schema_fau, 'fau')
  writer_fau = index_fau.writer()
  for (f,a),u in fau.items():
    u += ' ' + a
    f = f.decode('utf-8')
    a = a.decode('utf-8')
    u = u.decode('utf-8')
    writer_fau.add_document(func_id = f, arg = a, utter = u)
  writer_fau.commit()

  print 'Done'
