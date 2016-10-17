"""
Creates NL index for F's and FA's using Whoosh.
  - F: functions
  - FA: function-argument

Requires that api.sqlite3 exists in this folder, and that it contains `f` and
`fa` tables. Use consolidate.py for generating those.

Usage:
    python index_nl.py
"""
import os
import sqlite3
import whoosh
import whoosh.fields as wf
import whoosh.index

INDEX_DIR = '../demo/index'
DB_FILE = 'api.sqlite3'

if __name__ == '__main__':
  # Create indices
  if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)

  print 'Reading database from %s'%DB_FILE

  with sqlite3.connect(DB_FILE) as db:
    cursor = db.cursor()
    cursor.execute('SELECT func_id, utter from fu')
    rows_fu = cursor.fetchall()

    cursor.execute('SELECT func_id, arg, utter from fau')
    rows_fau = cursor.fetchall()

  print 'Creating fu NL index'

  schema_fu = wf.Schema(func_id=wf.ID(stored=True), utter=wf.TEXT(stored=True))
  index_fu = whoosh.index.create_in(INDEX_DIR, schema_fu, 'fu')
  writer_fu = index_fu.writer()
  for f,u in rows_fu:
    writer_fu.add_document(func_id = f, utter = u)
  writer_fu.commit()

  print 'Creating fau NL index'

  schema_fau = wf.Schema(func_id=wf.ID(stored=True), arg=wf.ID(stored=True), utter=wf.TEXT(stored=True))
  index_fau = whoosh.index.create_in(INDEX_DIR, schema_fau, 'fau')
  writer_fau = index_fau.writer()
  for f,a,u in rows_fau:
    writer_fau.add_document(func_id = f, arg = a, utter = u)
  writer_fau.commit()

  print 'Done'
