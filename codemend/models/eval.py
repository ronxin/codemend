"""
Evaluates baselines against the function and parameter highlighting task.

Steps:
1. Load the ground truth file.
2. Load the code samples.
3. Instantiate baseline callback methods.
4. For each code sample,
   - supply code sample and queries to the callback method
   - collect results
   - calculate metrics
5. Output metrics as a csv file.

"""

import ast
import csv
import numpy as np
import random
from collections import namedtuple
from itertools import imap

from annotate_code_with_api import get_fu_fau, findCallNodes, extractCallComponents
from whoosh_baseline import NLQueryBaseline
from baseline import RandomBaseline
from word2vec_baseline import Word2vecBaseline
from bimodal_baseline import BiModalBaseline

class EvalBrain:
  def __init__(self):
    self.gts = []
    # load the ground-truth file
    with open('eval-gt.csv', 'rb') as csv_file:
      reader = csv.reader(csv_file)
      # columns:
      # case_study_no,answer_func,answer_arg,query
      GroundTruth = namedtuple('GroundTruth', next(reader))
      for gt in imap(GroundTruth._make, reader):
        self.gts.append(gt)

    # load the code samples
    self.code_samples = []
    fnames = ['../demo/code-samples/eval%d.py'%x for x in [1,2,3,4,5]]
    for f in fnames:
      with open(f) as reader:
        code = reader.read().strip()
        self.code_samples.append(code)

    # load the function-utterance mappings for function names
    fu, fau = get_fu_fau()
    self.fs = set(fu.keys())
    self.fas = set(fau.keys())

    self.current_code_sample = None
    self.current_node = None

    # instantiate the baselines
    print 'Instantiating baselines...'
    self.baselines = []

    whoosh_baseline = NLQueryBaseline()  # keep a handle, for closing later

    self.baselines.append(whoosh_baseline)
    self.baselines.append(RandomBaseline())

    print 'Loading word2vec baselines, will take a while ...'

    w2v_mpl_as_term = Word2vecBaseline('output/vectors-flat-mpl-0205.bin')
    w2v_mpl_as_seq = Word2vecBaseline(w2v_mpl_as_term.model, fu_fau=(fu,fau))
    self.baselines.append(w2v_mpl_as_term)
    self.baselines.append(w2v_mpl_as_seq)

    w2v_py_as_term = Word2vecBaseline('output/vectors-flat-python427MB-0205.bin')
    w2v_py_as_seq = Word2vecBaseline(w2v_py_as_term.model, fu_fau=(fu,fau))
    self.baselines.append(w2v_py_as_term)
    self.baselines.append(w2v_py_as_seq)


    w2v_py5g_as_term = Word2vecBaseline('output/vectors-so-text-python-5gram.bin',
                                        maxngram=5)
    w2v_py5g_as_seq = Word2vecBaseline(w2v_py5g_as_term.model,
                                       fu_fau=(fu,fau),
                                       maxngram=5)
    self.baselines.append(w2v_py5g_as_term)
    self.baselines.append(w2v_py5g_as_seq)

    w2v_py3g_as_seq = Word2vecBaseline('output/vectors-so-text-python-stem-3gram.bin',
                                       fu_fau=(fu,fau),
                                       maxngram=3,
                                       use_stem=True)
    self.baselines.append(w2v_py3g_as_seq)

    """
    print 'Loading bimodal baselines, will take a while ...'
    self.baselines.append(BiModalBaseline('output/model-0204_with_args'))
    self.baselines.append(BiModalBaseline('output/model-0205-k100-iter100_with_args'))
    self.baselines.append(BiModalBaseline('output/model-0205-multi-k100-iter100-with_args'))
    """

    # evaluate the baselines
    print 'Starts evaluating...'
    metric_names = ['MRR_func', 'MRR_arg']
    results = np.zeros((len(self.baselines), len(metric_names)), dtype=float)
    result_log = []  # for diagnosis
    ResultLogEntry = namedtuple('ResultLogEntry',
                                ['code_sample_idx',
                                 'query',
                                 'func',  # only populated for args
                                 'baseline',
                                 'candidate',
                                 'rank',
                                 'is_gt',  # 1 or 0
                                 'score',
                                 'num_alt', # number of alternatives
                                ])
    count_func = 0
    count_arg = 0
    for idx, code_sample in enumerate(self.code_samples):
      # NOTE: The outer loop must be code samples. Because bimodal_baseline is
      # depending on the correct self.current_node. So, do NOT change this
      # triple-for-loop structure: {code-sample -> gt -> baseline}.
      print 'Processing code sample %d'%(idx + 1)
      self.current_code_sample = code_sample
      self.current_node = ast.parse(code_sample)
      current_gts = filter(lambda x: int(x.case_study_no) == int(idx + 1), self.gts)
      assert current_gts
      called_funcs, called_args = self.get_called_func_arg_lists()
      assert len(called_funcs) > 1
      for gt in current_gts:  # "gt" = "ground truth"
        count_func += 1
        assert gt.answer_func in called_funcs, gt.answer_func
        for b_idx, b in enumerate(self.baselines):
          called_funcs_shuffled = list(called_funcs)
          random.shuffle(called_funcs_shuffled)
          sorted_funcs_raw = b.rank_funcs(gt.query, called_funcs_shuffled, self)
          sorted_funcs = [x[0] for x in sorted_funcs_raw]
          assert gt.answer_func in sorted_funcs

          # Update MRR_func
          answer_rank = sorted_funcs.index(gt.answer_func) + 1
          mrr_idx = metric_names.index('MRR_func')
          results[b_idx, mrr_idx] += 1. / answer_rank

          # Update result log
          for cand_idx, cand in enumerate(sorted_funcs_raw):
            is_gt = 1 if cand[0] == gt.answer_func else 0
            score = cand[1] if len(cand) > 1 else 0
            result_log.append(ResultLogEntry(idx+1, gt.query, None, str(b),
                                             cand[0], cand_idx+1, is_gt, score,
                                             len(called_funcs)))

        if gt.answer_arg and str(gt.answer_arg)[0] not in '0123456789':
          if gt.answer_func in called_args and len(called_args[gt.answer_func]) > 1:
            count_arg += 1
            current_called_args = called_args[gt.answer_func]
            assert gt.answer_arg in current_called_args
            for b_idx, b in enumerate(self.baselines):
              current_called_args_shuffled = list(current_called_args)
              sorted_args_raw = b.rank_args(gt.query, gt.answer_func,
                                            current_called_args_shuffled, self)
              sorted_args = [x[0] for x in sorted_args_raw]
              assert gt.answer_arg in sorted_args

              # update MRR_arg
              answer_rank = sorted_args.index(gt.answer_arg) + 1
              mrr_idx = metric_names.index('MRR_arg')
              results[b_idx, mrr_idx] += 1. / answer_rank

              # update result log
              for cand_idx, cand in enumerate(sorted_args_raw):
                is_gt = 1 if cand[0] == gt.answer_arg else 0
                score = cand[1] if len(cand) > 1 else 0
                result_log.append(ResultLogEntry(idx+1, gt.query, gt.answer_func, str(b),
                                                 cand[0], cand_idx+1, is_gt, score,
                                                 len(current_called_args)))

    assert count_func > 0
    for metric_idx, metric in enumerate(metric_names):
      if metric.endswith('_func'):
        results[:,metric_idx] /= count_func
      else:
        assert metric.endswith('_arg')
        if count_arg > 0:
          results[:, metric_idx] /= count_arg

    # output
    print 'Writing outputs...'
    with open('output/eval-result-0211.csv', 'wb') as csv_file:
      writer = csv.writer(csv_file)
      writer.writerow(['Baseline'] + metric_names)
      for b_idx, b in enumerate(self.baselines):
        writer.writerow([b.__repr__()] + results[b_idx].tolist())

    with open('output/eval-log-0211.csv', 'wb') as csv_file:
      writer = csv.writer(csv_file)
      writer.writerow(ResultLogEntry._fields)
      for row in result_log:
        writer.writerow(row)

    # close resources
    print 'Closing resources'
    whoosh_baseline.close()

    print 'Done'

  def get_called_func_arg_lists(self):
    """
    Returns two lists:
     - called_funcs: [func_name1, func_name2, ...]
     - called_args: [func_name] = [arg1, arg2, ...]

    Only the function names and keywords arguments that occur in the mined API
    docs are listed.

    """
    assert self.current_node
    # NOTE: setting call_nodes as a member variable is important here, as
    # individual baseline methods may use the call_nodes to directly get
    # function-call relations.
    self.call_nodes = findCallNodes(self.current_node)
    called_funcs = []
    called_args = {}
    for call in self.call_nodes:
      func_name, keywords = extractCallComponents(call)
      if func_name in self.fs:
        called_funcs.append(func_name)
        called_args[func_name] = []
        for k in keywords:
          if (func_name, k) in self.fas:
            called_args[func_name].append(k)
    return called_funcs, called_args

if __name__ == '__main__':
  eb = EvalBrain()
