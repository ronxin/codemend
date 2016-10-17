"""
Simplified BiModal model.

"""

from __future__ import division
import numpy as np
from numpy import float32 as REAL
import multiprocessing as mp
from functools import partial
from gensim.utils import SaveLoad

from codemend.models.coke import get_lineno_elems_list
from codemend.models.nl_util import tokenize, lemmaWithVocab, stopwords
from codemend.models.ngram_util import ngram_partition
from codemend.docstring_parse import doc_serve

def train_example(model, code):
  global count_examples

  with count_examples.get_lock():
    count_examples.value += 1
    count_examples_val = count_examples.value
    alpha = model.alpha * (1 - count_examples_val / (model.total_train_examples+1))
    if alpha < model.alpha * 0.0001:
      alpha = model.alpha * 0.0001
    if count_examples_val % 500 == 0:
      print count_examples_val, '/', model.total_train_examples, 'alpha =', alpha

  np.random.seed(count_examples_val)

  lineno_elem_list = get_lineno_elems_list(code)
  if not lineno_elem_list: return
  used_elems = set([e for elist in lineno_elem_list for e in elist])
  used_elem_idxs = set([model.elem_lookup[ex] for ex in used_elems if ex in model.elem_lookup])
  for line_x, elist_x in enumerate(lineno_elem_list):

    if not elist_x: continue

    # Context
    p_idxs = []
    start = max(0, int(line_x - model.window / 2))
    end = min(len(lineno_elem_list), int(line_x + model.window / 2) + 1)
    for line_y in xrange(start, end):
      if np.random.random() > 0.5: continue  # randomly drop context lines

      elist_y = lineno_elem_list[line_y]
      for ey in elist_y:
        if np.random.random() > 0.5: continue  # randomly drop context elements

        if not ey in model.elem_lookup: continue
        eyi = model.elem_lookup[ey]
        p_idxs.append(eyi)

    p_idxs = list(set(p_idxs))

    # Target
    for ex in elist_x:
      if model.rand_parent_doc:
        with_parents = np.random.random() > 0.5
      else:
        with_parents = False
      doc = doc_serve.get_training_doc(ex, with_parents)
      if not doc: continue

      l_idxs = get_l_idxs(model, doc)

      if not ex in model.elem_lookup: continue
      r_pos_idx = model.elem_lookup[ex]

      r_neg_idxs = []
      cum_table = model.cum_table
      while len(r_neg_idxs) < model.negative:
        ex_neg = cum_table.searchsorted(np.random.randint(cum_table[-1]))
        if model.neg_sample_used_elem:
          if not ex.startswith(model.all_elems[ex_neg]):  # prevent adding parent as negative sample
            r_neg_idxs.append(ex_neg)
        else:
          if ex_neg not in used_elem_idxs:
            r_neg_idxs.append(ex_neg)

      update_params(model, p_idxs, l_idxs, r_pos_idx, r_neg_idxs, alpha)

def update_params(model, p_idxs, l_idxs, r_pos_idx, r_neg_idxs, alpha):
  global syn0p, syn1r, syn1b

  assert len(r_neg_idxs) > 0

  syn0l = model.syn0l  # fixed
  syn0p = np.ctypeslib.as_array(syn0p)
  syn1r = np.ctypeslib.as_array(syn1r)
  syn1b = np.ctypeslib.as_array(syn1b)

  if p_idxs:
    p_vecs = syn0p[p_idxs]
    p = p_vecs.mean(0)
  else:
    p = np.zeros(model.vector_size, dtype=REAL)

  if l_idxs:
    l_vecs = syn0l[l_idxs]
    l = l_vecs.mean(0)
  else:
    l = np.zeros(model.vector_size, dtype=REAL)

  if model.additive:
    h = p + l
  elif model.multiply:
    h = np.multiply(p, l)
  else:
    h = np.concatenate((p, l))

  r_indices = [r_pos_idx] + r_neg_idxs
  r_vecs = syn1r[r_indices]
  b = syn1b[r_indices]

  # f: outputs of the output-layer nodes
  f = 1. / (1. + np.exp(- np.dot(h, r_vecs.T) - b))  # 1 * (1 + negative)

  g = (model.neg_labels[:len(r_indices)] - f) * alpha  # 1 * (1 + negative)

  # gradient of likelihood wrt h times learning rate
  neu1e = np.dot(g, r_vecs)  # 1 * vector_size

  # learn hidden -> output weights
  syn1r[r_indices] += np.outer(g, h)

  # learn hidden -> output bias
  syn1b[r_indices] += g

  # gradient of likelihood wrt c times learning rate
  if model.additive:
    neu1e_c = neu1e
  elif model.multiply:
    neu1e_c = np.multiply(neu1e, l)
  else:
    neu1e_c = neu1e[:model.vector_size]

  # learn input vectors
  if p_idxs:
    syn0p[p_idxs] += neu1e_c / len(p_idxs)

  # Debug Only. Check f before vs. after updating
  # if model.all_elems[r_indices[0]] == 'plt.xlabel':
  #   print ' '.join([model.all_elems[i] for i in p_idxs])

  #   p_vecs = syn0p[p_idxs]
  #   p = p_vecs.mean(0)
  #   h = p + l
  #   r_vecs = syn1r[r_indices]
  #   b = syn1b[r_indices]
  #   f_after = 1. / (1. + np.exp(- np.dot(h, r_vecs.T) - b))
  #   print 'f_before', f
  #   print 'f_after', f_after
  #   print

  # global count_examples
  # if count_examples.value > 10:
  #   import sys
  #   sys.exit()


def get_l_idxs(model, utter):
  w2v_model = model.w2v_model
  tokens = tokenize(utter)

  if model.use_lemma:
    tokens = map(lambda x:lemmaWithVocab(x, w2v_model.vocab), tokens)

  if model.maxngram > 1:
    tokens_ngrams = ngram_partition(' '.join(tokens), w2v_model.vocab)
    tokens = list(set(tokens) | set(tokens_ngrams))

  tokens = filter(lambda x: x not in stopwords, tokens)

  idxs = [w2v_model.vocab[w].index for w in tokens if w in w2v_model.vocab]
  if not idxs:
    if w2v_model.null_word:
      idxs = [0]
    else:
      idxs = None
  return idxs


def _init(count_examples_, syn0p_, syn1r_, syn1b_):
  global count_examples, syn0p, syn1r, syn1b
  count_examples = count_examples_
  syn0p = syn0p_
  syn1r = syn1r_
  syn1b = syn1b_


class BiModal(SaveLoad):
  def __init__(self, all_elems, all_elems_counts, w2v_model, code_examples,
               enormer, negative=5, epoch=1, seed=1, alpha=0.1,
               additive=0, concat=1, multiply=0,
               window=20, use_lemma=True, maxngram=1, threads=None,
               rand_parent_doc=False, hint_pvecs_init=True, hint_rvecs_init=False,
               neg_sample_used_elem=False):
    """
    Parameters
    ----------
    - enormer: element normalizer
    - code_examples: generator of code example
    - window: number of lines in the coocurrence window
    - hint_pvecs_init: initialize code context vectors with NL weights
    - hint_rvecs_init: initialize code production vectors with NL weights

    """
    self.all_elems = all_elems
    self.all_elems_counts = all_elems_counts
    self.w2v_model = w2v_model
    self.code_examples = code_examples
    self.enormer = enormer
    self.negative = negative
    self.epoch = epoch
    self.seed = seed
    self.alpha = alpha
    self.additive = int(additive)
    self.concat = int(concat)
    self.multiply = int(multiply)
    self.window = window
    self.use_lemma = use_lemma
    self.maxngram = maxngram
    self.threads = threads
    self.rand_parent_doc = rand_parent_doc
    self.hint_pvecs_init = hint_pvecs_init
    self.hint_rvecs_init = hint_rvecs_init
    self.neg_sample_used_elem = neg_sample_used_elem

    if additive + concat + multiply != 1:
      raise ValueError('(additive + concat + multiply) must equal to 1')

    self.vector_size = self.w2v_model.vector_size
    self.elem_lookup = dict((elem, i) for (i, elem) in enumerate(all_elems))

    self.init_weights()
    self.train()

  def init_weights(self):
    self.syn0l = self.w2v_model.syn0
    self.syn0p = np.empty((len(self.all_elems), self.vector_size), dtype=REAL)
    r_vec_size = self.vector_size * 2 if self.concat else self.vector_size
    self.syn1r = np.empty((len(self.all_elems), r_vec_size), dtype=REAL)
    self.syn1b = np.empty(len(self.all_elems), dtype=REAL)

    sum_count = sum(self.all_elems_counts.values())
    for i in xrange(len(self.all_elems)):
      if self.hint_pvecs_init:
        self.syn0p[i] = self.get_init_vector_by_doc(self.all_elems[i])
      else:
        self.syn0p[i] = self.seeded_vector(self.all_elems[i] + str(self.seed))

      if self.hint_rvecs_init:
        if self.concat:
          r_vec_half = self.get_init_vector_by_doc(self.all_elems[i])
          self.syn1r[i] = np.concatenate((r_vec_half, r_vec_half))
        else:
          self.syn1r[i] = self.get_init_vector_by_doc(self.all_elems[i])
      else:
        self.syn1r[i] = self.seeded_vector('syn1r' + self.all_elems[i] + str(self.seed),
                                           r_vec_size)

      self.syn1b[i] = np.log(self.all_elems_counts[self.all_elems[i]] / sum_count)

  def get_init_vector_by_doc(self, elem):
    utter = doc_serve.get_training_doc(elem, True)
    l_idxs = get_l_idxs(self, utter)
    if l_idxs:
      l_vecs = self.syn0l[l_idxs]
      l = l_vecs.mean(0)
    else:
      l = np.zeros(self.vector_size, dtype=REAL)
    return l

  def seeded_vector(self, seed_string, vector_size=None):
    """Create one 'random' vector (but deterministic by seed_string)"""
    if not vector_size:
      vector_size = self.vector_size
    once = np.random.RandomState(hash(seed_string) & 0xffffffff)
    return (once.rand(vector_size) - 0.5) / vector_size

  def train(self):
    self.neg_labels = np.zeros(self.negative + 1)
    self.neg_labels[0] = 1.
    self.total_train_examples = sum(1 for _ in self.code_examples()) * self.epoch
    self.make_cum_table()

    count_examples = mp.Value('i', 0)

    def make_shared(arr):
      return mp.sharedctypes.RawArray(arr._type_, arr)

    syn0p = make_shared(np.ctypeslib.as_ctypes(self.syn0p))
    syn1r = make_shared(np.ctypeslib.as_ctypes(self.syn1r))
    syn1b = make_shared(np.ctypeslib.as_ctypes(self.syn1b))
    initargs = (count_examples, syn0p, syn1r, syn1b)

    if self.threads is None or self.threads > 1:
      pool = mp.Pool(processes=self.threads, initializer=_init,
        initargs=initargs)

      examples = list(self.code_examples())
      for ep in xrange(self.epoch):
        pool.map(partial(train_example, self), examples)

    else:
      _init(*initargs)
      for example in self.code_examples():
        train_example(self, example)

    self.syn0p = np.ctypeslib.as_array(syn0p)
    self.syn1r = np.ctypeslib.as_array(syn1r)
    self.syn1b = np.ctypeslib.as_array(syn1b)

    print 'Training finished.'


  def make_cum_table(self, power=0.75, domain=2**31 - 1):
    # Borrowed from gensim's word2vec's make_cum_table
    self.cum_table = {}

    vocab_size = len(self.all_elems)
    self.cum_table = np.zeros(vocab_size, dtype=np.uint32)

    # compute sum of all power (Z in paper)
    train_words_pow = float(sum([x**power for x in self.all_elems_counts.values()]))

    cumulative = 0.0
    for word_index in xrange(vocab_size):
      cumulative += self.all_elems_counts[self.all_elems[word_index]]**power / train_words_pow
      self.cum_table[word_index] = round(cumulative * domain)

    if len(self.cum_table) > 0:
      assert self.cum_table[-1] == domain

  def save(self, *args, **kwargs):
    kwargs['ignore'] = kwargs.get('ignore', ['cum_table, w2v_model, code_examples', 'syn0l'])
    super(BiModal, self).save(*args, **kwargs)
    print 'Model saved.'

  def score_all(self, utter, used_elems):
    """
    Returns an numpy array of relevancy scores for all elements.

    """
    utter = utter.lower()
    h = self.compute_h(utter, used_elems)
    return np.dot(h, self.syn1r.T) + self.syn1b

  def score(self, utter, used_elems, candidate_elems):
    """
    Returns a list of (elem, score) tuples.

    Those supplied candidate elements that do not exist in vocabulary will be
    excluded.

    """
    utter = utter.lower()
    h = self.compute_h(utter, used_elems)
    r_elems = filter(lambda x: x in self.elem_lookup, candidate_elems)
    if not r_elems: return []
    r_idxs = map(lambda x: self.elem_lookup[x], r_elems)
    scores = np.dot(h, self.syn1r[r_idxs].T) + self.syn1b[r_idxs]
    return zip(r_elems, scores)

  def compute_h(self, utter, used_elems):
    """
    Returns Computes hidden layer value.

    """
    p_idxs = set([self.elem_lookup[e] for e in used_elems if e in self.elem_lookup])
    if p_idxs:
      p_vecs = self.syn0p[list(p_idxs)]
      p = p_vecs.mean(0)
    else:
      p = np.zeros(self.vector_size, dtype=REAL)

    l_idxs = get_l_idxs(self, utter)
    if l_idxs:
      l_vecs = self.syn0l[l_idxs]
      l = l_vecs.mean(0)
    else:
      l = np.zeros(self.vector_size, dtype=REAL)

    if self.additive:
      h = p + l
    elif self.multiply:
      h = np.multiply(p, l)
    else:
      h = np.concatenate((p, l))
    return h
