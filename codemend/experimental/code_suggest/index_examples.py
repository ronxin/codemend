"""
Run this after score_examples.py.

Create an SQLite table with code examples, indexed by functions. Each code
example also has its corresponding generated SVGs.

The table is like this:
  (func_id, code, svg)

There are at most 20 (shortest) examples per func_id.

"""
import sqlite3

from codemend import BackupHandler, relative_path

if __name__ == '__main__':

  print 'Reading SVGs and code examples. Takes 7.3 seconds...'
  bh = BackupHandler('.')
  svgs = bh.load('svgs')
  all_codes = bh.load('all_codes')
  plotcommands_examples = bh.load('plotcommands_examples')  # [plot_command] = [example_idx]

  db = sqlite3.connect(relative_path('demo/data/code.sqlite3'))
  cursor = db.cursor()

  cursor.executescript("""
    DROP TABLE IF EXISTS example;

    CREATE TABLE example (
      func_id TEXT NOT NULL,
      code TEXT NOT NULL,
      svg TEXT
    );

    CREATE INDEX func_id_idx ON example (func_id);
    """)

  for i, (func, example_idxs) in enumerate(plotcommands_examples.items()):
    print '%d / %d'%(i+1, len(plotcommands_examples))
    # The example_idxs have already been sorted (by their length) in
    # score_examples.py
    example_idxs = example_idxs[:20]
    for idx in example_idxs:
      code = all_codes[idx]
      svg = svgs[idx]
      assert code
      assert svg
      svg = svg.decode('utf-8')
      cursor.execute("INSERT INTO example VALUES (?, ?, ?)", (func, code, svg))

  db.commit()
  db.close()
