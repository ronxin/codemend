"""
@Deprecated. Use mine_element.py instead.

Extracts counts of:
 - functions
 - args per func
 - value per (func, arg)

Current limitation: only looking at the following value types:
 - ast.Str
 - ast.Num
 - ast.Name (including True, False)
 - ast.Tuple
 - ast.Dict
 - ast.List

Future work:
 - consider these following value types:
   - a variable assigned earlier as part of data
   - a variable assigned earlier by result of another function call
   - an inline function call
 - consider positional arguments of functions

Steps:
1. Load the collection of functions and arguments of interest.
2. Load all code examples.
3. For each code example,
  - parse it to obtain call statements
  - use the call statement to populate index (maintaining a count of value -
    of possible interest)
4. Use in-house element structure to index the counts.
5. Save the result as a pickle.

"""

import ast
import astunparse
from collections import defaultdict

from codemend import BackupHandler, relative_path
from codemend.demo.code_suggest import Element
from codemend.models.annotate_code_with_api import get_fu_fau, findCallNodes, extractCallComponents

def extractFAV(node):
  """
  Given an ast.Call node, return a tuple of (func, [keyword, value]).

  For value, only selected types are considered (see Current Limitation on
  top).

  Positional arguments are ignored for now.

  See also: annotate_code_with_api.extractCallComponents().

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
  kvs = []
  if hasattr(node, 'keywords'):
    for x in node.keywords:
      # x is a ast.keyword instance
      k = x.arg
      v = x.value
      if isinstance(v, ast.Num):
        kvs.append((k, repr(v.n)))
      elif isinstance(v, ast.Str):
        kvs.append((k, repr(v.s)))
      elif isinstance(v, ast.Name):
        kvs.append((k, v.id))
      elif isinstance(v, ast.Tuple) \
           or isinstance(v, ast.Dict) \
           or isinstance(v, ast.List):
        kvs.append((k, astunparse.unparse(v).strip()))
  return func_name, kvs


if __name__ == '__main__':

  # Step 1.
  fu, fau = get_fu_fau()

  # Step 2.
  bh = BackupHandler(relative_path('experimental/code_suggest'))
  all_codes = bh.load('all_codes')
  print 'There are %d code examples in total'%len(all_codes)

  # Step 3.
  f_counts = defaultdict(int)  # [f] = count
  fa_counts = defaultdict(int)  # [f,a] = count
  fav_counts = defaultdict(int)  # [f,a,v] = count
  for code in all_codes:
    try:
      node = ast.parse(code)
    except SyntaxError:
      continue
    calls = findCallNodes(node)

    for call in calls:

      f, ks = extractCallComponents(call)

      f_counts[f] += 1

      for k in ks:
        fa_counts[f,k] += 1

      f, kvs = extractFAV(call)
      for k, v in kvs:
        if not (f,k) in fau: continue
        fav_counts[f,k,v] += 1

  # Reindex fa_count by [f]
  f_a_counts = defaultdict(dict)  # [f][a] = count
  for f,a in fa_counts:
    f_a_counts[f][a] = fa_counts[f,a]
  f_a_counts = dict(f_a_counts)

  # Reindex fav_count by [f,a]
  fa_v_counts = defaultdict(dict)  # [f,a][v] = count
  for f,a,v in fav_counts:
    fa_v_counts[f,a][v] = fav_counts[f,a,v]
  fa_v_counts = dict(fa_v_counts)

  print 'Example f counts (plot)'
  print f_counts['plot']

  print 'Example fa counts:'
  print f_a_counts['plot']

  print 'Example fav counts:'
  print fa_v_counts['plot','color']
  print

  print 'There are %d keys in fa_v_counts'%len(fa_v_counts)

  # Prepare hierarchical suggestion index.
  element_index = {}  # [id] = element

  # (1). function as index
  for f, u in fu.items():
    count = f_counts[f] if f in f_counts else 0
    element_index[f] = Element(f, u, count, None)

  # (2). function, argument as index
  for (f, a), u in fau.items():
    if not f in element_index:
      count_f = f_counts[f] if f in f_counts else 0
      element_index[f] = Element(f, '', count_f, None)
    count = fa_counts[f, a] if (f,a) in fa_counts else 0
    element_index[f, a] = Element(a, u, count, element_index[f])

  # (3). add children to function, argument
  # Coming from experimental/code_suggest/mine_argvs.py
  for (f, a), v_counts in fa_v_counts.items():
    if not (f, a) in element_index: continue
    for v, count in v_counts.items():
      element_index[f, a, v] = Element(v, '', count, element_index[f, a])

  # (4). Sort all children by count then by value
  for elem in element_index.values():
    elem.children = sorted(elem.children, key=lambda x: (-x.count, x.val))

  print '%d total entries in element index'%len(element_index)

  bh2 = BackupHandler(relative_path('demo/data'))
  bh2.save('element_index', element_index)

  """
There are 15770 code examples in total
Example f counts (plot)
4228
Example fa counts:
{'mfc': 28, 'xlim': 1, 'markeredgewidth': 11, 'markeredgecolor': 20,
'linewidth': 169, 'rot': 4, 'style': 20, 'layout': 1, 'lc': 1, 'title': 14,
'lw': 183, 'ls': 34, 'yerr': 5, 'markersize': 117, 'grid': 1, 'xdata': 1,
'ys': 1, 'rasterized': 3, 'drawstyle': 2, 'x_compat': 2, 'dashes': 3, 'x': 29,
'picker': 13, 'edgecolor': 2, 'table': 3, 'edge_labels': 1, 'whis': 1, 'zs':
11, 'latlon': 3, 'sharey': 1, 'sharex': 2, 'markerfacecolor': 25, 'label':
527, 'colormap': 4, 'mec': 19, 'mew': 19, 'antialiased': 3, 'sym': 1,
'startangle': 1, 'legend': 17, 'c': 112, 's': 2, 'markeresize': 1, 'autopct':
1, 'clip_on': 25, 'color': 526, 'xerr': 2, 'scaley': 1, 'visible': 6,
'marker': 191, 'xs': 1, 'markeredecolor': 1, 'transform': 24, 'xticks': 6,
'width': 4, 'gid': 1, 'linestlye': 3, 'zdir': 17, 'ydata': 1, 'kind': 105,
'xunits': 1, 'stacked': 10, 'error_kw': 2, 'zorder': 59, 'ms': 61, 'animated':
5, 'aa': 2, 'figure': 1, 'ax': 91, 'linestyle': 117, 'ylim': 4, 'axis': 1,
'secondary_y': 5, 'markevery': 4, 'fillstyle': 3, 'graph_border': 1,
'fontsize': 2, 'solid_capstyle': 7, 'figsize': 5, 'yticks': 1, 'lab': 2,
'alpha': 78, 'url': 1, 'subplots': 5, 'y': 24, 'position': 2}

Example fav counts:
{"'purple'": 1, '(1, 0, 0, 0.5)': 1, "'#aaaaff'": 2, "'g--'": 1, 'color': 30,
"'Blue'": 1, '(0.0, 0.5, T[i])': 1, "'#ee8d18'": 2, 'colour': 1, "'0.2'": 1,
'colors': 4, "['red', 'blue', 'green']": 1, "'#00ff00'": 1, "'pink'": 2,
'blue': 1, '(0, 1, 0)': 1, "'c'": 3, "'grey'": 3, 'COLOR': 3, "'none'": 1,
'velocity_color': 1, "'#FF4455'": 1, '[1, 0, 0, 0.2]': 1, "'r'": 58,
"'#3F7F4C'": 1, "'lightblue'": 1, "'#1B2ACC'": 1, "'0.75'": 2, '(1, 0, 0)': 1,
"['r', 'b', 'b', 'b']": 1, "'red'": 54, "'b'": 39, "'y'": 3, "'0.5'": 3,
"'green'": 25, 'my_colors': 1, "'.3'": 2, "'#80C0FF'": 1, "'blue'": 40,
"'forestgreen'": 2, '(r, g, b)': 1, '[1, 0, 0]': 1, 'c': 14, "['r', 'g',
'b']": 1, "'g'": 22, "'#CC4F1B'": 1, "'#0066FF'": 1, "'yellow'": 8, "'black'":
31, "'gray'": 3, "'#3399FF'": 1, "'#AA5555'": 1, "'brown'": 1, 'acolor': 1,
"'w'": 4, '((i / float(n)), ((i / float(n)) ** 4), ((i / float(n)) ** 2))': 1,
"'aqua'": 1, "'.75'": 5, "[sns.xkcd_rgb['brownish red']]": 1, 'color_list': 1,
"'#112233'": 1, "'#b9cfe7'": 2, 'colorVal': 1, '(0, 0, 1)': 3, "'k'": 43,
"'white'": 1, "'0.1'": 1, '(0.4, 0.5, 0.6)': 1, 'col': 2, "'cornflowerblue'":
1}

There are 624 keys in fa_v_counts
9060 total entries in element index
Saved to /Users/ronxin/Dropbox/git/codemend/codemend/demo/data/element_index.pickle
  """
