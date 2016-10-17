"""
Read code examples from supp_examples_0229.ipynb and use them to update
pickles of svgs and all_codes, so that all plotting comamnds have at least one
code example.

Run this after mine_examples.py.
After this, run score_examples.py and index_examples.py.

"""
import matplotlib
matplotlib.use('Agg')  # prevents window from popping up
import json
import ast
import StringIO

import sys
sys.path.append('../../models')
from annotate_code_with_api import findCallNodes, extractCallComponents

from backup_util import BackupHandler


SUPP_FUNCS = ['angle_spectrum', 'barbs', 'cohere', 'csd', 'eventplot', 'magnitude_spectrum',
 'phase_spectrum', 'plotfile', 'psd', 'spy', 'streamplot', 'tricontour', 'violinplot',
 'xcorr']

if __name__ == '__main__':
  bh = BackupHandler('.')
  old_svgs = bh.load('svgs')
  old_all_codes = bh.load('all_codes')

  with open('supp_examples_0229.ipynb') as reader:
    notebook = reader.read()

  func_code_idx = {}  # [func_name] = code_idx
  codes = []
  svgs = []
  cells = json.loads(notebook)['cells']
  for cell in cells:
    if cell['cell_type'] == 'code':
      if 'outputs' in cell and cell['outputs']:
        code = ''.join(cell['source'])
        node = ast.parse(code)
        calls = findCallNodes(node)
        for call in calls:
          func_name, keywords = extractCallComponents(call)
          if func_name in SUPP_FUNCS and func_name not in func_code_idx:
            func_code_idx[func_name] = len(codes)
            codes.append(code)

            # Get SVG
            imgdata = StringIO.StringIO()
            codeObj = compile(code, '<string>', 'exec')
            exec codeObj
            plt.savefig(imgdata, format='svg', bbox_inches='tight')
            plt.close()
            imgdata.seek(0)
            svg = imgdata.buf
            svgs.append(svg)

  svgs = old_svgs + svgs
  all_codes = old_all_codes + codes
  bh.save('svgs', svgs)
  bh.save('all_codes', all_codes)
