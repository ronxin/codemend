"""
Mine elements (as defined in models) from code examples.

Count occurrences of each element.

"""

import ast
from collections import defaultdict

from codemend import BackupHandler, relative_path
from codemend.models.element import Element, ElementNormalizer
from codemend.models.element_extract import extract_varmap_elems

all_codes1 = []
all_codes2 = []
all_codes3 = []
def code_examples():
  """
  Yield code examples.

  """

  global all_codes1, all_codes2, all_codes3

  # 15770 code examples mined from SO answers in threads that are tagged
  # "matplotlib".
  if not all_codes1:
    print 'Loading SO code examples...'
    bh1 = BackupHandler(relative_path('experimental/code_suggest'))
    all_codes1 = bh1.load('all_codes')
    print '%d examples from SO'%len(all_codes1)

  for code in all_codes1:
    yield code

  # print 'WARNING: mine_element.py ignoring all GitHub code examples...'
  # """
  if not all_codes2:
    # 8732 code examples (including 395 IPython Notebook files) mined from
    # GitHub repositories that contain "matplotlib".
    print 'Loading GitHub code examples...'
    bh2 = BackupHandler(relative_path('experimental/code_suggest/output/backup'))
    all_codes2 = bh2.load('all_codes_github_1k_repo_0322')
    print '%d examples from GitHub'%len(all_codes2)

  for code in all_codes2:
    yield code
  # """

  if not all_codes3:
    # 21993 code examples extracted by Shiyan from the Web
    print 'Loading Web code examples'
    bh3 = BackupHandler(relative_path('experimental/mining/output'))
    all_codes3 = bh3.load('codes_shiyan_0331_web')
    print '%d examples from Web Shiyan'%len(all_codes3)

  for code in all_codes3:
    yield code

def get_countable_value(v, varmap, enormer):
  """
  Returns a string, which is the repr of the value, or elem_id if it is a
  variable. Or may be more complex when it is a tuple or a list.

  """
  if not v: return
  if isinstance(v, ast.Num):
    return repr(v.n)
  elif isinstance(v, ast.Str):
    return repr(v.s)
  elif isinstance(v, ast.Name):
    id_ = v.id
    if id_ in ('True', 'False'):
      return id_
    elif id_ in varmap:
      return '$' + enormer.simplify(varmap[id_])
  elif isinstance(v, ast.Tuple) or isinstance(v, ast.List):
    if v.elts:
      elt_strs = []
      for e in v.elts:
        e_str = get_countable_value(e, varmap, enormer)
        if e_str:
          elt_strs.append(e_str)
      if elt_strs:
        elts_concat = ', '.join(elt_strs)
        fmt = '(%s)' if isinstance(v, ast.Tuple) else '[%s]'
        return fmt%elts_concat

if __name__ == '__main__':
  enormer = ElementNormalizer()
  counters = defaultdict(int)
  element_counts = defaultdict(int)
  element_pyplot_counts = defaultdict(int)  # [elem] = count
  element_pyplot_value_counts = defaultdict(lambda: defaultdict(int))  # [elem][val] = count
  count = 0
  for code in code_examples():
    count += 1
    if count % 1000 == 0: print '%d ...'%count
    try:
      varmap, elems = extract_varmap_elems(code)
    except SyntaxError:
      counters['synax_error_files'] += 1
      continue
    except TypeError as e:
      print code
      raise e

    for e in elems:
      elem_id = e.elem
      element_counts[elem_id] += 1

      elem_id = enormer.simplify(elem_id)
      if elem_id.startswith('plt.'):
        element_pyplot_counts[elem_id] += 1
        val = get_countable_value(e.val_node, varmap, enormer)
        if val: element_pyplot_value_counts[elem_id][val] += 1

  for elem_id in element_pyplot_value_counts:
    element_pyplot_value_counts[elem_id] = dict(element_pyplot_value_counts[elem_id])
  element_pyplot_value_counts = dict(element_pyplot_value_counts)

  print 'Processed %d code examples'%count
  print 'There are %d unique elements'%len(element_counts)
  print 'There are %d unique pyplot elements'%len(element_pyplot_counts)
  for k in counters:
    print '%s: %d'%(k, counters[k])

  bh = BackupHandler(relative_path('experimental/code_suggest/output/backup'))
  # Change logs:
  # - 0322: using raw format
  # - 0327: using Element, tracking return type and variable assignments and
  #   import aliases.
  # - 0404: fixed issue with dict as positional argument;
  #         added element_value_counts;
  #         added Shiyan's example.
  bh.save('elem_counts_0404', element_counts)
  bh.save('elem_pyplot_counts_0404', element_pyplot_counts)
  bh.save('elem_pyplot_value_counts_0404', element_pyplot_value_counts)

  """
  Log:

  # 0327
  Processed 24502 code examples
  There are 144898 unique elements
  There are 7741 unique pyplot elements
  Saved to /Users/ronxin/Dropbox/git/codemend/codemend/experimental/code_suggest/output/backup/elem_counts_0327.pickle
  Saved to /Users/ronxin/Dropbox/git/codemend/codemend/experimental/code_suggest/output/backup/elem_pyplot_counts_0327.pickle
  synax_error_files: 3223

  # 0404
  Processed 46495 code examples
  There are 177033 unique elements
  There are 9569 unique pyplot elements
  synax_error_files: 3223
  Saved to /Users/ronxin/Dropbox/git/codemend/codemend/experimental/code_suggest/output/backup/elem_counts_0404.pickle
  Saved to /Users/ronxin/Dropbox/git/codemend/codemend/experimental/code_suggest/output/backup/elem_pyplot_counts_0404.pickle
  Saved to /Users/ronxin/Dropbox/git/codemend/codemend/experimental/code_suggest/output/backup/elem_pyplot_value_counts_0404.pickle

  """
