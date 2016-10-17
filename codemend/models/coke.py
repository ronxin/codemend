"""
Utilities for counting co-occurrence between elements.

"""

from collections import defaultdict

from codemend import BackupHandler, relative_path
from codemend.models.element import ElementNormalizer
from codemend.models.element_extract import extract_varmap_elems
from codemend.experimental.code_suggest.mine_element import code_examples

enormer = ElementNormalizer()

def get_lineno_elems_list(code):
  try:
    _, elems = extract_varmap_elems(code)
  except SyntaxError:
    return

  # index elements by lineno
  lineno_elems = defaultdict(list)  # [lineno] = [elem]
  for e in elems:
    e_str = transform_and_filter(e.elem)
    if e_str: lineno_elems[e.lineno].append(e_str)

  if not lineno_elems:
    return

  max_lineno = max(lineno_elems.keys())
  lineno_elems_list = [lineno_elems[i] for i in xrange(max_lineno+1)]
  return lineno_elems_list

def get_cokes(code, window=20):
  """
  Yields (elem1, elem2).

  Always: elem1 < elem2.

  transform_and_filter() is applied internally.

  """
  lineno_elems_list = get_lineno_elems_list(code)
  if not lineno_elems_list: return

  # Generate co-occurrence
  for lx, elx in enumerate(lineno_elems_list):
    for ly in xrange(lx, lx + window):
      if ly >= len(lineno_elems_list): break
      ely = lineno_elems_list[ly]
      for ex in elx:
        for ey in ely:
          if ex == ey: continue
          elif ex < ey: yield ex, ey
          else: yield ey, ex

def transform_and_filter(elem):
  """
  Cleaning is performed to reduce sparsity:
    - pylab.xxx --> plt.xxx (if the function exists in pyplot)
    - various add_subplot.xxx --> plt.gca.xxx (see stype.tsv)
    - only plt.* are kept

  Returns: cleaned elem or None

  """
  elem = enormer.simplify(elem)
  if elem.startswith('plt.'):
    return elem
  else:
    return None

if __name__ == '__main__':
  coke_counts = defaultdict(int)
  count = 0
  for code in code_examples():
    count += 1
    if count % 1000 == 0:
      print '%d ... unique_cokes=%d'%(count, len(coke_counts))

    for x, y in get_cokes(code):
      coke_counts[x,y] += 1

  bh = BackupHandler(relative_path('models/output/backup'))
  bh.save('coke_0329', coke_counts)
