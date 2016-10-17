"""
Annotate code with API documentations.

1. Read API documentations to obtain mapping between function, argument, and
   natural language.
   - Only the actual name of the function is used to identify a function. The
     module portion is discarded.
   - If there are multiple NL's mapped to a function (due to that the same
     function name occurs in different packages), then
     - pick the most popular NL description.
     - If there is a tie, then use both/all NL descriptions by concatenating
       them.
   - (The API documents are tokenized but NOT lower-cased.) Make all
     utterances lower-cased.

2. For the code blocks extracted from Stack Overflow:
   - get separated code blocks by splitting using '\\n\\n\\n'
   - for each code block,
     - clean it to replace HTML escaped characters
     - try using ast to parse it. Discard it if parsing fails.
     - extract all top-level function calls
     - try looking up the function (and optionally arguments) in the extracted
       API mappings.
     - concatenate the found NL utterances if there are more than one of them
     - save the (utter, code) tuple as one training instance.
"""

import ast
import astunparse  # easy_install astunparse
import csv
import os
from collections import defaultdict

from codemend import BackupHandler, relative_path

def get_most_popular_lowered(counter):
  """
  Given a counter: [key]=count

  Returns the most popular key. If there is a tie, then return a concatenation
  of all the keys that have the largest count.
  """
  max_count = max(counter.values())
  keys = [k for k,v in counter.items() if v == max_count]
  s = ' '.join(keys)
  s = s.lower()
  return s

def get_fu_fau(omit_module=True, truncate=True):
  """
  Do step 1.

  Returns fu, fau
  fu: [func_name] = most_popular_utter
  fau: [func_name, arg] = most_popular_utter

  Parameters
  ----------
  omit_module: if True, the func_name will be the last part only.

  """

  def get_func_name(fullName):
    if omit_module:
      return fullName.split('.')[-1]
    else:
      return fullName


  fu = defaultdict(lambda: defaultdict(int))  # [func_name][utter] = count
  with open(relative_path('docstring_parse/fu.csv'), 'rb') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the header
    for f, u in reader:
      u = u.split('|||')[0]  # see consolidate.py for meaning of |||
      if truncate: u = ' '.join(u.split()[:15])  # limit the maximum number of tokens
      fu[get_func_name(f)][u] += 1

  fau = defaultdict(lambda: defaultdict(int))  # [func_name, arg][utter] = count
  with open(relative_path('docstring_parse/fau.csv'), 'rb') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the header
    for f, a, u in reader:
      u = u.split('|||')[0]  # see consolidate.py for meaning of |||
      if truncate: u = ' '.join(u.split()[:15])  # limit the maximum number of tokens
      fau[get_func_name(f), a][u] += 1

  # consolidate the fu and fau mappings
  for f in fu:
    fu[f] = get_most_popular_lowered(fu[f])
  fu = dict(fu)

  for fa in fau:
    fau[fa] = get_most_popular_lowered(fau[fa])
  fau = dict(fau)

  return fu, fau

def findCallNodes(node):
  """
  Given an ast node.

  Returns: a list of ast.Call nodes.

  NOTE: Call nodes that are descendants of another call node are ignored. For
  example, in the following code:
      plt.gca().set_title(get_str())
  Only the function call corresponding to "set_title" is returned.

  """
  class MyVisitor(ast.NodeVisitor):
    def __init__(self):
      self._call_nodes = []

    def visit_Call(self, node):
      self._call_nodes.append(node)

    def get_call_nodes(self):
      return self._call_nodes

  v = MyVisitor()
  v.visit(node)
  return v.get_call_nodes()


def extractCallComponents(node):
  """
  Given an ast.Call node, return a tuple of (func_name, keywords)
  """
  assert isinstance(node, ast.Call)
  assert hasattr(node, 'func')
  func = node.func
  if isinstance(func, ast.Attribute):
    assert hasattr(func, 'attr')
    func_name = func.attr
  elif isinstance(func, ast.Name):
    assert hasattr(func, 'id')
    func_name = func.id
  else:
    # something like: f()(), e.g., strpdate2num(datefmt)(x)
    func_name = None
  if hasattr(node, 'keywords'):
    keywords = [x.arg for x in node.keywords]
  else:
    keywords = []
  return func_name, keywords


def get_train_pairs(fu, fau, blocks, include_arguments):
  total_block = 0
  total_grammatical = 0
  total_call_nodes = 0
  total_matched_funcs = 0
  total_matched_args = 0

  train_pairs = []
  for i,block in enumerate(blocks):
    if (i + 1) % 1000 == 0: print '%d / %d'%(i+1, len(blocks))
    total_block += 1
    try:
      node = ast.parse(block)
      total_grammatical += 1
    except SyntaxError:
      continue

    call_nodes = findCallNodes(node)
    for call_node in call_nodes:
      total_call_nodes += 1

      f, args = extractCallComponents(call_node)
      if not f or not f in fu:
        continue

      total_matched_funcs += 1

      f_utter = fu[f]
      assert f_utter, f

      if include_arguments:
        arg_utters = [fau[f,a] for a in args if (f,a) in fau]
        total_matched_args += len(arg_utters)
      else:
        arg_utters = []

      merged_utter = ' '.join([f_utter] + arg_utters)

      train_pairs.append((merged_utter, astunparse.unparse(call_node)))

  unique_train_pairs = list(set(train_pairs))

  print 'total_block', total_block
  print 'total_grammatical', total_grammatical
  print 'total_call_nodes', total_call_nodes
  print 'total_matched_funcs', total_matched_funcs, '(total train pairs)'
  print 'total_matched_args', total_matched_args
  print 'total_unique_train_pairs', len(unique_train_pairs)

  return unique_train_pairs


if __name__ == '__main__':
  bh = BackupHandler(relative_path('models/output/backup'))

  # Step 1
  fu, fau = get_fu_fau()

  # Step 2
  with open(relative_path('models/output/mpl_code_blocks.txt')) as reader:
    content = reader.read()

  content = content.decode('utf-8')
  content = content.replace("&lt;", "<")
  content = content.replace("&gt;", ">")
  content = content.replace("&amp;", "&")

  blocks = content.split('\n\n\n')

  assert len(blocks) > 100

  train_pairs = get_train_pairs(fu, fau, blocks, include_arguments=False)
  bh.save('train_pairs_0204', train_pairs)

  train_pairs_with_args = get_train_pairs(fu, fau, blocks, include_arguments=True)
  bh.save('train_pairs_0204_with_args', train_pairs_with_args)
