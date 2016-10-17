"""
Mine code frequency information from github code repositories.

1. Take all code from GitHub.
2. Do MD5 hashing of files to avoid duplicates.
3. Filter files to only keep files that use matplotlib or pylab.
4. Parse codes to ASTs.
5. Collect parameters statistics and value statistics for a fixed set of
   functions.
6. All unique parseable codes are saved for other purposes (e.g.,
   mine_element.py). IPython notebook cells within the same notebook are
   concatenated as one block.


TODO:
 - expand the support set of values
 - support variable assignment tracking -- so as to know their types

See also: mine_argvs.py

"""

import ast
import md5
import json
import os
from collections import defaultdict

from codemend import BackupHandler, relative_path
from codemend.models.annotate_code_with_api import get_fu_fau, findCallNodes
from codemend.experimental.code_suggest.mine_argvs import extractFAV

def usesMatplotlib(code):
  """
  Returns True if the code appears to be using matplotlib.

  """
  if 'from matplotlib' in code: return True
  if 'import matplotlib' in code: return True
  if 'from pylab' in code: return True
  if 'import pylab' in code: return True
  return False

def parseAndCount(code, elem_counts, fu):
  node = ast.parse(code)
  calls = findCallNodes(node)
  is_useful = False
  for call in calls:
    f, avs = extractFAV(call)
    if f in fu:
      is_useful = True
    else:
      continue
    elem_counts[f,] += 1
    for a, v in avs:
      elem_counts[f,a] += 1
      elem_counts[f,a,v] += 1
  return is_useful

if __name__ == '__main__':
  counters = defaultdict(int)

  md5s = set()

  all_codes = []

  fu, _ = get_fu_fau()

  elem_counts = defaultdict(int)  # [elem] = count

  bh = BackupHandler(relative_path('experimental/code_suggest/output/backup'))

  for root, dirs, files in os.walk(
      relative_path('mining/output/github-matplotlib-repos')):

    if '.git' in root: continue

    for file_name in files:
      counters['count_file'] += 1

      if counters['count_file'] % 1000 == 0:
        print 'Processed %d files - Useful files: %d'%(
          counters['count_file'], counters['count_useful_files'])

      file_path = os.path.join(root, file_name)

      try:
        with open(file_path) as reader:
          code = reader.read().decode('utf-8')
      except (UnicodeEncodeError, UnicodeDecodeError) as e:
        counters['count_encoding_error'] += 1
        # print '%s for %s'%(e, file_path)
        continue
      except IOError:
        counters['count_io_error'] += 1
        continue

      # Hash file name and code to dedupe
      hash_ = md5.new(file_name + code.encode('utf-8')).digest()
      if hash_ in md5s:
        counters['count_duplicates'] += 1
        continue
      else:
        md5s.add(hash_)

      # Throw away if not using matplotlib
      if not usesMatplotlib(code):
        counters['count_not_use_matplotlib'] += 1
        continue

      is_useful = False  # has at least one valid call
      if file_name.endswith('.py'):
        try:
          is_useful = parseAndCount(code, elem_counts, fu)
        except SyntaxError:
          counters['count_not_parseable_py'] += 1
          continue
        if is_useful:
          counters['count_useful_py_file'] += 1
        all_codes.append(code)

      elif file_name.endswith('.ipynb'):
        notebook_code_cells = []
        try:
          nb = json.loads(code)
        except ValueError:
          counters['bad_notebook_json'] += 1
          continue
        if 'cells' in nb:
          cells = nb['cells']
        elif 'worksheets' in nb:
          cells = nb['worksheets'][0]['cells']

        for cell in cells:
          if cell['cell_type'] == 'code':
            if 'source' in cell:
              src = ''.join(cell['source'])
            else:
              src = ''.join(cell['input'])
            if not src:
              continue
            try:
              is_useful |= parseAndCount(src, elem_counts, fu)
            except SyntaxError:
              counters['count_not_parseable_cell'] += 1
              continue
            notebook_code_cells.append(src)

        if is_useful:
          counters['count_useful_notebook_file'] += 1

        if notebook_code_cells:
          all_codes.append('\n'.join(notebook_code_cells))

      else:
        counters['count_bad_suffix'] += 1

      if is_useful:
        counters['count_useful_files'] += 1

  bh.save('elem_counts_0322', elem_counts)
  for cnt_key in sorted(counters.keys()):
    print '%s: %d'%(cnt_key, counters[cnt_key])

  bh.save('all_codes_github_1k_repo_0322', all_codes)
