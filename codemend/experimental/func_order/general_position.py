"""
Extract general positions of matplotlib API functions
1. for a code example, extract all matplotlib API functions
2. label each function's position as the percentile among all API functions in
   this code
3. take average per function

Output:
 - a dictionary: [function] = average_position
   average position: between 0 (beginning of code) and 1 (end of code).
"""

import ast
from collections import defaultdict

from codemend import BackupHandler, relative_path
from codemend.models.annotate_code_with_api import get_fu_fau, findCallNodes, extractCallComponents

fu, fau = get_fu_fau()
bh = BackupHandler(relative_path('experimental/code_suggest'))
all_codes = bh.load('all_codes')
print 'There are %d code examples in total'%len(all_codes)

pos_sum = defaultdict(float)  # [f] = sum
pos_cnt = defaultdict(int)  # [f] = count
for code in all_codes:
  try:
    node = ast.parse(code)
  except SyntaxError:
    continue
  calls = findCallNodes(node)
  called_funcs = [extractCallComponents(x)[0] for x in calls]
  called_funcs = filter(lambda x: x in fu, called_funcs)
  if len(calls) < 3:
    continue
  for i, f in enumerate(called_funcs):
    pos_sum[f] += float(i) / len(called_funcs)
    pos_cnt[f] += 1

pos_ave = {}
for f in pos_sum:
  pos_ave[f] = pos_sum[f] / pos_cnt[f]

print 'Extracted average positions for %d functions'%len(pos_ave)

bh2 = BackupHandler(relative_path('demo/data'))
bh2.save('pos_ave', pos_ave)
