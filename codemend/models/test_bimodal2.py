from __future__ import division

import ast
import csv

from codemend import relative_path
from codemend.models.bimodal2 import BiModal
from codemend.models.word2vec_util import load_gensim_from_binary_file
from codemend.models.constraint import prune
from codemend.models.baseline2 import SuggestItem
from codemend.thonny import ast_utils
from codemend.models.eval2 import ContextBuilder

small_gt = [
  # gt, query, used_elems
  ('plt.bar@color', 'change bar color', ['plt.bar', 'plt.bar@color', 'plt.xlabel']),
  ('plt.title', 'add title', ['plt.bar', 'plt.bar@color', 'plt.xlabel']),
  ('plt.xlabel', 'add label to x-axis', ['plt.bar', 'plt.bar@color', 'plt.xlabel']),
  ('plt.pie', 'create a pie chart', []),
  ('plt.bar@hatch', 'add hatching', ['plt.bar', 'plt.bar@hatch', 'plt.title']),
  ('plt.ylabel', '', ['plt.bar', 'plt.xlabel', 'plt.title']),
  ('plt.title@bbox@facecolor', 'change color of title background', ['plt.bar', 'plt.title']),
  ('plt.barh@hatch', 'make a horizontal bar plot . set the hatching pattern plt barh hatch', ['plt.barh']),
  ('plt.legend@shadow', 'add shadow', ['plt.plot', 'plt.legend']),
  ('plt.grid', 'add grid lines', ['plt.pie']),
]

small_gt2 = [
  # gt, query, file
  ('plt.bar@color', 'change bar color', 'bar'),
  ('plt.title', 'add title', 'bar'),
  ('plt.xlabel', 'add label to x-axis', 'bar'),
  ('plt.bar@hatch', 'add hatching', 'bar'),
  ('plt.bar@hatch', 'fill pattern', 'bar'),
  # ('plt.bar@hatch', 'add bar marks', 'bar'),
  ('plt.ylabel', '', 'bar'),
  ('plt.title@bbox@facecolor', 'change color of title background', 'pie'),
  ('plt.pie@explode', 'explode', 'pie'),
  ('plt.pie@explode', 'separate a piece of pie', 'pie'),
  ('plt.pie@explode', 'explode pie', 'pie'),
  ('plt.title@bbox', 'title bounding box', 'pie'),
  ('plt.title@bbox@pad', 'title bounding box padding', 'pie'),
  ('plt.title@bbox@pad', 'title size', 'pie'),
  ('plt.title@bbox@facecolor', 'title background color', 'pie'),
  ('plt.grid', 'add grid line', 'bar'),
  ('plt.grid', 'add grid lines', 'bar'),
  ('plt.grid', 'add grid', 'bar'),
  ('plt.grid', 'add grids', 'bar'),
  ('plt.grid', 'add gridlines', 'bar'),
  ('plt.plot@linewidth', 'thickness', 'line'),
  ('plt.plot@linewidth', 'line thickness', 'line'),
  ('plt.plot@linewidth', 'thick', 'line'),
  ('plt.plot@linewidth', 'wide', 'line'),
  # ('plt.xticks@rotation', 'change the style of x-axis label', 'bar'),
  ('plt.xkcd', 'fancy style', 'line'),
]

gt_set = set(small_gt2)

small_gt3 = []
with open(relative_path('models/data/gt-0924.csv'), 'rb') as csvfile:
  reader = csv.reader(csvfile)
  next(reader, None)  # skip the header
  for file_,query,expected,remark in reader:
    combined = (expected, query, file_)
    if combined not in gt_set:
      small_gt3.append(combined)
    else:
      print 'duplicated: %s %s %s'%combined

cb = ContextBuilder()

def anything_to_used_elems(anything):
  if isinstance(anything, basestring):
    filename = {
      'bar': relative_path('demo/code-samples/user-study/task1.py'),
      'pie': relative_path('demo/code-samples/user-study/task2.py'),
      'line': relative_path('demo/code-samples/user-study/practice.py'),
      'empty': relative_path('demo/code-samples/empty.py'),
      'eval3': relative_path('demo/code-samples/eval3.py'),
      'line_video': relative_path('demo/code-samples/demo_video_linechart.py')
    }[anything]
    with open(filename) as reader:
      code = reader.read()
    if not code.strip(): return []
    node = ast.parse(code)
    ast_utils.mark_text_ranges(node, unicode(code))
    context = cb.getContext(node)
    return context.used_elems()
  elif isinstance(anything, list):
    return anything
  else:
    raise TypeError(type(anything))

def eval_one(model, with_pruning, gt, q, anything):
  used_elems = anything_to_used_elems(anything)
  scores = model.score_all(q, used_elems)
  elems_sorted = sorted(zip(scores, model.all_elems), reverse=True)
  suggest_sorted = [SuggestItem(elem=elem, score=score) for (score, elem) in elems_sorted]
  if with_pruning:
    suggest_sorted = prune(used_elems, suggest_sorted)
  tmps = []  # elements ranked before gt
  rank = 0
  for i, suggest in enumerate(suggest_sorted):
    if i < 10:
      tmps.append(suggest.elem)
    if suggest.elem == gt:
      rank = i + 1
      break
  return rank, tmps, suggest_sorted

def eval_batch(model, with_pruning):
  all_ranks = []
  qs = []
  mrr = 0.
  perfect_ranks = 0
  good_ranks = 0
  for gt, q, anything in small_gt + small_gt2 + small_gt3:
    qs.append(q)
    rank, _, _ = eval_one(model, with_pruning, gt, q, anything)
    all_ranks.append(rank)
    if rank > 0: mrr += 1 / rank
    if rank == 1: perfect_ranks += 1
    if rank > 0 and rank <= 5: good_ranks += 1
  mrr /= len(all_ranks)
  q_rank = sorted(zip(all_ranks, qs), key=lambda x: (x[0]==0,x[0]), reverse=True)

  print
  print '[Mean Rank]: %f'%(sum(all_ranks) / len(all_ranks))
  print '[MRR]: %f'%(mrr)
  print '[Perfect Ranks]: %f'%(perfect_ranks / len(all_ranks))
  print '[Good Ranks]: %f'%(good_ranks / len(all_ranks))
  print
  print 'Worst ranks:'
  for rank, q in q_rank[:15]:
    print 'Rank: %5d   %s'%(rank, q)
  print
  print 'Best ranks:'
  count = 0
  for rank, q in reversed(q_rank):
    count += 1
    if count > 15: break
    print 'Rank: %5d   %s'%(rank, q)

def load_model(model_id=None):
  # model_file_name = 'models/output/bi2-test.model'
  model_file_name = 'models/output/bi2-0410-d.model'
  if model_id:
    print 'Using customized model_id'
    model_file_name = 'models/output/bi2-0410' + model_id + '.model'

  w2v_model_file = 'vectors-so-text-python-lemma.bin'
  if model_id == '-s':
    w2v_model_file = 'vectors-so-text-python-lemma-win3.bin'
  elif model_id == '-t':
    w2v_model_file = 'vectors-so-text-python-lemma-win5.bin'
  w2v_model = load_gensim_from_binary_file(
    relative_path('models/output/' + w2v_model_file))

  model = BiModal.load(relative_path(model_file_name))
  print "@@@ PLEASE CHECK WHICH FILE IS BEING TESTED ... @@@"
  print '@@@ MODEL_FILE: %s @@@'%model_file_name

  model.w2v_model = w2v_model
  model.syn0l = w2v_model.syn0
  return w2v_model, model

if __name__ == '__main__':
  import sys
  model_id = None
  if len(sys.argv) >= 2:
    model_id = sys.argv[1]

  w2v_model, model = load_model(model_id)

  print
  print
  print '@@@@  NO PRUNING  @@@@'
  eval_batch(model, with_pruning=False)

  print
  print
  print '@@@@  WITH PRUNING  @@@@'
  eval_batch(model, with_pruning=True)
