The order of running those files is:

- `mine_examples.py`: take in code blocks previously extracted from matplotlib-
  tagged SO posts. Use multiprocessing to execute and obtain svgs.
- `add_supp_examples.py`: take in manually-written ipynb for those plotting
  commands that do not have executable code examples. Generates svgs and
  update the output (pickle) of mine_examples.py. These code examples are
  largely based on matplotlib's official code demos.
- `score_examples.py`: index code examples by plotting commands, and sort the
  examples by effective length (number of characters in lines that do not have
  import statements). The output is plotcommands_examples.pickle.
- `index_examples.py`: create a SQLite table with (func, code, svg) fields.


**Note**: the above files cannot be executed until adapted to the new hierarchy caused
by repackaging on Mar 02, 2016.

