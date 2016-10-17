import os
import ast
from collections import defaultdict

from codemend import relative_path, BackupHandler
from codemend.experimental.code_suggest.mine_argvs_github import usesMatplotlib

root = '/storage6/users/shiyansi/codemend/results/'

counters = defaultdict(int)
count = 0
codes = []
for filename in os.listdir(root):
  count += 1
  if count % 1000 == 0: print count
  counters['1. files'] += 1
  path = os.path.join(root, filename)
  with open(path) as reader:
    content = reader.read()

  if not usesMatplotlib(content):
    counters['2. files not using matplotlib'] += 1
    continue

  blocks = content.split('\n\n\n')

  for block in blocks:
    counters['3. blocks'] += 1

    lines = block.split('\n')

    if len(lines) <= 3:
      counters['4. too short blocks'] += 1
      continue

    try:
      ast.parse(block)
      counters['6. blocks parseable'] += 1
      codes.append(block)

    except SyntaxError:
      counters['5. blocks with syntax error'] += 1

for key in sorted(counters.keys()):
  print '%s: %d'%(key, counters[key])

bh = BackupHandler(relative_path('experimental/mining/output'))
bh.save('codes_shiyan_0331_web', codes)

print 'Done'
