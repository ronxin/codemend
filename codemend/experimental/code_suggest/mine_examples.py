"""
Mine code examples from the following soures:
 - matplotlib cookbook code examples
 - stackoverflow posts tagged matplotlib

For each top-level plotting function (taken from pyplot summary page), find a
set of code examples, such that:
 - scored and ordered by the number of function calls in the code
 - at most 10 examples per plotting function
 - must generate an non-empty SVG

Pipeline:
 1. start by copying the step 2 of annotate_code_with_api.py
    - the mpl_code_blocks is only 4.5 MB (really really small)
 2. add code examples from the textbook to our collection
 3. for each code block, treat it as a code example
    - clean it
    - try to parse it (throw away if unparseable)
    - try to execute it (throw away if unevalable)
 4. save all good code examples in a pickle
    (Next: run score_examples.py)
"""
import __builtin__
import string
import multiprocessing
from functools import partial
from multiprocessing.pool import ThreadPool
from pylab import *

from codemend import BackupHandler

punctuations = ''.join(x for x in string.punctuation if x != '_')
replace_punctuation = string.maketrans(punctuations, ' '*len(punctuations))

def issafe(code):
  """
  A fairly weak code safety checker.
  """
  if '__' in code:
    return False

  if isinstance(code, unicode):
    code = code.encode('utf-8')

  code = code.translate(replace_punctuation)
  code_tokens = code.split()

  code_token_set = set(code_tokens)

  if 'sys' in code_token_set: return False
  if 'os' in code_token_set: return False
  if 'savefig' in code_token_set: return False
  if 'sleep' in code_token_set: return False
  if 'raw_input' in code_token_set: return False
  if 'SystemExit' in code_token_set: return False
  if 'py2exe' in code_token_set: return False
  if 'multiprocessing' in code_token_set: return False
  if 'threading' in code_token_set: return False
  if 'write_png' in code_token_set: return False

  return True

def get_svg(code, counters, send_end):
  """
  Worker function.

  Updates counters.
  Sends the resulted svg via send_end.

  """
  import ast
  import numpy as np
  import matplotlib
  import matplotlib.pyplot as plt
  import matplotlib.pyplot as PLT
  import pylab
  import StringIO
  import string

  # Parse it
  try:
    node = ast.parse(code)
  except SyntaxError:
    counters['syntax_errors'].increment()
    send_end.send(None)
    return

  # Safety check:
  if not issafe(code):
    counters['unsafes'].increment()
    send_end.send(None)
    return

  # Evaluate it
  try:
    imgdata = StringIO.StringIO()
    codeObj = compile(code, '<string>', 'exec')
    exec codeObj
  except Exception:
    counters['exec_errors'].increment()
    send_end.send(None)
    return

  # Check if any figure is produced at all
  import matplotlib.pyplot as plt
  if not plt.get_fignums():
    counters['nofigures'].increment()
    send_end.send(None)
    return

  # Save figure as SVG
  try:
    plt.savefig(imgdata, format='svg', bbox_inches='tight')
  except:
    counters['savefig_errors'].increment()
    send_end.send(None)
    return

  plt.close()
  imgdata.seek(0)
  svg = imgdata.buf
  if __builtin__.len(svg) <= 700:
    # 630 is an empty figure on Surong's iMac
    counters['empty_svgs'].increment()
    send_end.send(None)
    return

  send_end.send(svg)
  counters['successes'].increment()


# http://stackoverflow.com/questions/31255118/python-process-pool-with-timeout-on-each-process-not-all-of-the-pool
def run_with_timeout(timeout, func, counters, *args):
  receive_end, send_end = multiprocessing.Pipe(duplex=False)
  p = multiprocessing.Process(target=func, args=args, kwargs=dict(
    send_end=send_end, counters=counters))
  p.daemon = True
  p.start()
  send_end.close()  # child must be the only one with it opened
  p.join(timeout)
  if p.is_alive():
    counters['timeouts'].increment()
    p.terminate()
  else:
    try:
      return receive_end.recv()  # get value from the child
    except EOFError:
      pass


import shared_counter

if __name__ == '__main__':
  import matplotlib
  matplotlib.use('Agg')  # prevents window from popping up

  # Step 1:
  # Copied from annotate_code_with_api.py
  with open('../../models/output/mpl_code_blocks.txt') as reader:
    content = reader.read()

  content = content.decode('utf-8')
  content = content.replace("&lt;", "<")
  content = content.replace("&gt;", ">")
  content = content.replace("&amp;", "&")

  sompl_blocks = content.split('\n\n\n')  # stackoverflow matplotlib code blocks
  print 'There are %d code examples from mpl stackoverflow'%len(sompl_blocks)

  # Step 2:
  bh = BackupHandler('.')
  cookbook_segs = bh.load('cookbook_segs')
  cookbook_blocks = []
  for tag, p in cookbook_segs:
    if tag == 'CODE':
      cookbook_blocks.append(p)

  print 'There are %d code examples from matplotlib cookbook'%len(cookbook_blocks)

  all_codes = sompl_blocks + cookbook_blocks

  print 'There are %d code blocks in total'%(len(all_codes))

  # Step 3:
  counters = {}
  counter_names = ['syntax_errors', 'unsafes', 'timeouts', 'exec_errors',
                   'nofigures', 'savefig_errors', 'empty_svgs', 'successes']
  for name in counter_names:
    counters[name] = shared_counter.Counter(name=name)

  pool = ThreadPool(processes=4)

  # all_codes = all_codes[:1000]  # DEBUG

  svgs = pool.map(partial(run_with_timeout, 3, get_svg, counters), all_codes)

  for counter in counters.values():
    print counter

  bh.save('svgs', svgs)
  bh.save('all_codes', all_codes)

  # LOG:
  # There are 15582 code examples from mpl stackoverflow
  # Restored from ./cookbook_segs.pickle
  # There are 174 code examples from matplotlib cookbook
  # There are 15756 code blocks in total
  # timeouts: 514
  # empty_svgs: 92
  # unsafes: 1420
  # exec_errors: 6165
  # savefig_errors: 68
  # successes: 2582
  # syntax_errors: 3223
  # nofigures: 1691
  # Saved to ./svgs.pickle
  # Saved to ./all_codes.pickle
