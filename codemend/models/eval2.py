"""
Evaluates baselines against Benchmark 2: suggesting modifications of AST.

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

import csv
import numpy as np
from recordclass import recordclass
from itertools import imap

from codemend import BackupHandler, relative_path

from codemend.models.baseline2 import Baseline, Context, RandomBaseline, SuggestItem
from codemend.models.element import ElementNormalizer
from codemend.models.element_extract import extract_varmap_elems
from codemend.models.word2vec_baseline2 import Word2vecBaseline
from codemend.models.bimodal_baseline2 import BiModalBaseline
from codemend.models.bimodal2 import BiModal  # to make pickler happy

ResultLogEntry = recordclass('ResultLogEntry',
                              ['code_sample_idx',
                               'query',
                               'baseline',
                               'candidate',
                               'rank',
                               'is_gt',  # 1 or 0
                               'score',
                               'answer_rank',
                               'answer',
                              ])

class EvalBrain:
  def __init__(self):
    self.expected_set_cache = {}

    self.queries = []
    # load the ground-truth file
    with open('eval-gt-2.csv', 'rb') as csv_file:
      reader = csv.reader(csv_file)
      # columns:
      # case_study_no,answer_func,answer_arg,query
      Query = recordclass('Query', next(reader))
      for query in imap(Query._make, reader):
        assert query.case_study_no.startswith('example-')
        query.case_study_no = int(query.case_study_no.replace('example-', ''))
        query.answer = query.answer.strip()
        query.query = query.query.strip()
        query.query_source = query.query_source.strip()
        self.queries.append(query)

    print 'Loading the code samples...'
    self.code_samples = []
    fnames = [relative_path('demo/code-samples/before_afters/before%d.py'%x)
              for x in [1,2,3,4,5]]
    for f in fnames:
      with open(f) as reader:
        code = reader.read().strip()
        self.code_samples.append(code)

    print 'Initializing context builder...'
    self.cb = ContextBuilder()

    print 'Initializing element normalizer...'
    self.enormer = ElementNormalizer()

    print 'Instantiating baselines...'
    self.baselines = []
    self.baselines.append(RandomBaseline(self.cb.getAllElements()))

    w2vb1 = Word2vecBaseline(
        relative_path('models/output/vectors-so-text-python-lemma.bin'),
        self.cb.getAllElementCounts(), 1, 'w2v')

    w2vb2 = Word2vecBaseline(w2vb1.model,
        self.cb.getAllElementCounts(), 1, 'w2v-heuristic', heuristic=True)

    w2vb3 = Word2vecBaseline(w2vb1.model,
        self.cb.getAllElementCounts(), 1, 'w2v-cooccur', use_coke=True)

    w2vb4 = Word2vecBaseline(w2vb1.model,
        self.cb.getAllElementCounts(), 1, 'w2v-hc', heuristic=True, use_coke=True)

    self.baselines += [w2vb1, w2vb2, w2vb3, w2vb4]

    # bimodal = BiModalBaseline('bimodal-concat-10epoch',
    #     relative_path('models/output/bi2-test-ggg.model'),
    #     relative_path('models/output/vectors-flat-mpl-0205.bin'))
    # bimodal_ids = list('denopq')
    bimodal_ids = list('d')
    for id_  in bimodal_ids:
      bimodal = BiModalBaseline('bimodal-'+id_,
          relative_path('models/output/bi2-0410-%s.model'%id_),
          w2vb1.model)
      self.baselines.append(bimodal)

    print 'Starts evaluating...'
    metric_names = ['MRR', 'P@1', 'P@5', 'P@10']
    results = np.zeros((len(self.baselines), len(metric_names)), dtype=float)
    result_log = []  # for diagnosis

    count_query = 0
    for idx, code in enumerate(self.code_samples):
      # triple-for-loop structure: {code-sample -> gt -> baseline}.
      print 'Processing code sample %d'%(idx + 1)

      current_queries = filter(lambda x: int(x.case_study_no) == int(idx + 1), self.queries)
      assert current_queries

      context = self.cb.getContext(code)

      for query in current_queries:  # "query" = "ground truth"
        count_query += 1
        assert query.answer

        for b_idx, b in enumerate(self.baselines):
          suggested_items = b.suggest(query.query, context)
          answer_rank = self.getRankOfExpectedItem(
              suggested_items, code, query.answer)

          mrr_idx = metric_names.index('MRR')
          p1_idx = metric_names.index('P@1')
          p5_idx = metric_names.index('P@5')
          p10_idx = metric_names.index('P@10')
          if answer_rank > 0:
            results[b_idx, mrr_idx] += 1. / answer_rank
            if answer_rank == 1:
              results[b_idx, p1_idx] += 1
            if answer_rank <= 5:
              results[b_idx, p5_idx] += 1
            if answer_rank <= 10:
              results[b_idx, p10_idx] += 1


          self.updateResultLog(result_log, idx + 1, query.query, b,
                               suggested_items, code, query.answer,
                               answer_rank)

    assert count_query > 0
    for metric_idx, metric in enumerate(metric_names):
      if metric == 'MRR' or metric.startswith('P@'):
        results[:, metric_idx] /= count_query

    # output
    print 'Writing outputs...'
    with open(relative_path('models/output/eval-result-0413.csv'), 'wb') as csv_file:
      writer = csv.writer(csv_file)
      writer.writerow(['Baseline'] + metric_names)
      for b_idx, b in enumerate(self.baselines):
        writer.writerow([b.__repr__()] + results[b_idx].tolist())

    with open(relative_path('models/output/eval-log-0413.csv'), 'wb') as csv_file:
      writer = csv.writer(csv_file)
      writer.writerow(ResultLogEntry._fields)
      for row in result_log:
        writer.writerow(row)

    # close resources
    print 'Closing resources'
    # whoosh_baseline.close()

    print 'Done'

  def getExpectedSet(self, code, answer):
    """
    Parse a hand-written answer and return an expected set.

    - TODO: support root type (treat it as a disambiguation problem)
    - TODO: support return type (also kind of a disambiguation problem)
    - TODO: support dictionary value (more levels in expected_set)

    """
    cache_key = (code, answer)
    if not cache_key in self.expected_set_cache:
      answer = answer.strip()
      answer_elems = answer.split()
      expected_set = set()
      for elem in answer_elems:
        elem = self.enormer.simplify(elem)
        expected_set.add(elem)

      self.expected_set_cache[cache_key] = expected_set

    return self.expected_set_cache[cache_key]

  def getRankOfExpectedItem(self, items, code, answer):
    """
    Scans items and look for the expected answer.

    Returns the highest rank of an item that matches the given answer.
       Return 0 if no match is found.

    """
    expected_set = self.getExpectedSet(code, answer)

    for i, item in enumerate(items):
      assert isinstance(item, SuggestItem), type(item)

      # This is already simplified by context builder
      # - Everything in and out of the model is simplified
      elem = item.elem
      if elem in expected_set:
        return i + 1

    return 0

  def updateResultLog(self, result_log, code_sample_idx, query, baseline,
                      items, code, answer, answer_rank):
    """
    Modifies result_log in-place.

    """
    expected_set = self.getExpectedSet(code, answer)

    for idx, item in enumerate(items):
      assert isinstance(item, SuggestItem)
      assert isinstance(item.elem, basestring)
      assert isinstance(query, basestring)
      assert isinstance(baseline, Baseline), baseline
      rank = idx + 1
      is_gt = int(item.elem in expected_set)
      result_log.append(ResultLogEntry(code_sample_idx, query, repr(baseline),
        item.elem, rank, is_gt, item.score, answer_rank, answer))


class ContextBuilder:
  def __init__(self):
    bh = BackupHandler(relative_path('experimental/code_suggest/output/backup'))
    elem_counts = bh.load('elem_pyplot_counts_0404')
    self.all_elems = set(elem_counts.keys())
    self.all_elem_counts = elem_counts
    self.enormer = ElementNormalizer()

  def getContext(self, code):
    var_map, elems = extract_varmap_elems(code)
    for elem in elems:
      elem.elem = self.enormer.simplify(elem.elem)
    elem_ids = [x.elem for x in elems]
    return Context(elem_ids, elems, var_map)

  def getAllElements(self):
    return self.all_elems

  def getAllElementCounts(self):
    return self.all_elem_counts

if __name__ == '__main__':
  eb = EvalBrain()
