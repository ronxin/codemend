"""
Run this script after mine_examples.py and add_supp_examples.py.
After this, run index_examples.py.

 1. Read code examples and their corresponding SVGs from a pickle.
 2. For each good code example
    - check which top level pyplot function is used (limiting to plotting
      functions)
    - score it
    - create an index by top level pyplot function
 3. for each top level pyplot function
    - go from the highest scored code example
    - if can generate non-empty SVG, add to output
    - stop after hitting a max limit (e.g., 20 examples) per function
 4. Saved the indexed code examples as a pickle.
"""
from codemend import BackupHandler


def get_effective_code_len(code):
  """
  Number of characters in a code example. Not counting lines with "import"

  """
  lines = code.split('\n')
  lines = filter(lambda x: 'import' not in x.split(), lines)
  return len('\n'.join(lines))


if __name__ == '__main__':

  print 'Reading SVGs and code examples. Takes 7.3 seconds...'
  bh = BackupHandler('.')
  svgs = bh.load('svgs')
  all_codes = bh.load('all_codes')




  print 'Loading functions that are plotting commands'
  # Copied from code_suggest.py
  import csv
  import pattern.en
  # Load csv file of pyplot summary
  pyplot_fu = {}  # [func] = utter
  print 'CodeSuggest: Loading pyplot fu...'
  with open('../../docstring_parse/pyplot_fu.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the header
    for f, u in reader:
      if not u:
        continue
      pyplot_fu[f] = u
  print 'CodeSuggest: read %d fu pairs'%len(pyplot_fu)
  # lowercase and tokenization of u's
  for f in pyplot_fu:
    pyplot_fu[f] = ' '.join(pattern.en.tokenize(pyplot_fu[f].lower()))
  # Extract plotting commands
  PLOT_PREFIXES = ['plot', 'make', 'draw']  # happens to be 4 chars
  plot_commands = [f for f,u in pyplot_fu.items() if u[:4] in PLOT_PREFIXES]
  plot_commands = sorted(plot_commands)
  print 'CodeSuggest: extracted %d plot commands'%len(plot_commands)




  print 'Indexing code examples'
  import sys
  sys.path.append('../../models')
  from annotate_code_with_api import findCallNodes, extractCallComponents
  import ast
  from collections import defaultdict
  plot_commands_set = set(plot_commands)
  examples = defaultdict(set)  # [plot_command] = [example_idx]
  seen_code_set = set()
  count_dupe = 0
  for i in xrange(len(all_codes)):
    if not svgs[i]: continue
    code = all_codes[i].strip()

    if code in seen_code_set:
      # Dedupe
      count_dupe += 1
      continue
    else:
      seen_code_set.add(code)

    node = ast.parse(code)
    calls = findCallNodes(node)
    for call in calls:
      func_name, keywords = extractCallComponents(call)
      if func_name in plot_commands_set:
        examples[func_name].add(i)

  print 'There are %d duplicates'%count_dupe



  print '"Scoring" code examples.'
  # Sorting function: number of chars in the code example
  examples = dict(examples)
  for func, idxs in examples.items():
    examples[func] = sorted(idxs, key=lambda x: get_effective_code_len(all_codes[x]))


  bh.save('plotcommands_examples', examples)
