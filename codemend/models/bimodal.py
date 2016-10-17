#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An implementation of the model presented in

"Bimodal Modelling of Source Code and Natural Language"
ICML 2016
Miltiadis Allamanis, Daniel Tarlow, Andrew Gordon, and Yi Wei

Implementation is heavily borrowed from gensim
https://github.com/piskvorky/gensim
"""
from __future__ import division  # py3 "true division"

import logging
import sys
import os
import heapq
from timeit import default_timer
from copy import deepcopy
from collections import defaultdict
import threading
import itertools

from gensim.utils import keep_vocab_item

try:
  from queue import Queue, Empty
except ImportError:
  from Queue import Queue, Empty

from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL
from numpy import uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis
from numpy import ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray
from numpy import multiply as np_mult, cumsum, argmax

from gensim import utils as gsutils, matutils  # utility fnc for pickling, common scipy operations etc
from six import iteritems, itervalues, string_types
from six.moves import xrange
from types import GeneratorType

import ast
import astunparse
from myast import MyAST, SimpleAstNode

logger = logging.getLogger(__name__)

try:
  from bimodal_inner import train_batch
  from bimodal_inner import FAST_VERSION, MAX_PTREES_IN_BATCH
except ImportError:
  # failed... fall back to plain numpy (20-80x slower training than the above)
  FAST_VERSION = -1
  MAX_PTREES_IN_BATCH = 10000
  NEG_SAMPLING_VOCAB_SIZE_THRESHOLD = 20  # minimum is 1

  def train_batch(model, train_pairs, alpha, work=None):
    """
    Update model by training on a sequence of train_pairs.

    Each train_pair is a pair of NL utterance and MyAST instance, which are
    looked up in the model's vocab dictionaries. Called internally from
    `BiModal.train()`.

    This is the non-optimized, Python version. If you have cython installed,
    then the optimized version from bimodal is used instead.
    """
    result = 0
    for utter, myast in train_pairs:
      l_idxs = get_l_idxs(model, utter)
      if not l_idxs: continue

      for ptree in myast.randomized_partial_trees(
            model.random, model.memsize_k, model.memsize_i):
        terminals, ancestors, parent, children = ptree

        k_idxs = [model.vocab_k[x.getSimple()] for x in reversed(terminals)]
        k_idxs += [0] * (model.memsize_k - len(k_idxs))  # padding

        i_idxs = [model.vocab_i[x.getSimple()] for x in ancestors]
        i_idxs += [0] * (model.memsize_i - len(i_idxs))  # padding

        parent = parent.getSimple()
        if isinstance(children, (list, tuple)):
          children = tuple(map(lambda x: x.getSimple(), children))
        else:
          children = children.getSimple()

        r_pos_idx, r_neg_idxs = find_r_idxs_train(model, parent, children)

        if not r_neg_idxs: continue  # only one possible children, meaningless to train.

        train_a_pair(model, i_idxs, k_idxs, l_idxs, r_pos_idx, r_neg_idxs, alpha)

        result += 1
    return result

def get_l_idxs(model, utter):
  l_tokens = utter.split()
  l_idxs = [model.vocab_l[w].index for w in l_tokens if w in model.vocab_l and
            # downsampling for handling top frequent words (stopwords)
            model.vocab_l[w].sample_int > model.random.rand() * 2**32]
  if not l_idxs:
    if model.null_word:
      l_idxs = [0]
    else:
      l_idxs = None
  return l_idxs

def find_r_idxs_train(model, parent, children):
  """
  Given a (parent, children) training instance, generate the positive children
  index by lookup and the negative children index by sampling. If the number
  of negative samples are too small then all of them are used in the parameter
  learning.
  """
  vocab = model.vocab_r[parent]
  index2word = model.index2word_r[parent]
  vocab_size = len(index2word)
  r_idx_offset = model.r_idx_offset[parent]

  idx = vocab[children].index
  r_pos_idx = idx + r_idx_offset

  if vocab_size <= model.negative + NEG_SAMPLING_VOCAB_SIZE_THRESHOLD:
    # The vocabulary is too small: we simply use all negative samples.
    r_neg_idxs = [x + r_idx_offset for x in range(vocab_size) if x != idx]
  else:
    cum_table = model.cum_table[parent]

    r_neg_idxs = []
    while len(r_neg_idxs) < model.negative:
      w = cum_table.searchsorted(model.random.randint(cum_table[-1]))
      if w != idx:
        r_neg_idxs.append(w + r_idx_offset)

  return r_pos_idx, r_neg_idxs

def find_r_idxs_sample(model, parent):
  """
  Returns the indexes of all possible children tuples under this parent.
  """
  index2word = model.index2word_r[parent]
  vocab_size = len(index2word)
  r_idx_offset = model.r_idx_offset[parent]
  return [x + r_idx_offset for x in range(vocab_size)]

def train_a_pair(model, i_idxs, k_idxs, l_idxs, r_pos_idx, r_neg_idxs, alpha):
  assert len(i_idxs) == model.hmati.shape[0]
  assert len(k_idxs) == model.hmatk.shape[0]
  assert len(l_idxs) > 0

  hmati = model.hmati  # memsize_i * vector_size
  hmatk = model.hmatk  # memsize_k * vector_size
  i_vecs = model.syn0i[i_idxs]  # memsize_i * vector_size
  k_vecs = model.syn0k[k_idxs]  # memsize_k * vector_size
  l_vecs = model.syn0l[l_idxs]  # arbitrary * vector_size

  c = np_mult(i_vecs, hmati).sum(0) + \
      np_mult(k_vecs, hmatk).sum(0)  # 1 * vector_size
  l = l_vecs.mean(0)  # 1 * vector_size

  if model.additive:
    h = c + l  # 1 * vector_size
  else:
    h = np_mult(c, l)

  r_indices = [r_pos_idx] + r_neg_idxs
  r_vecs = model.syn1r[r_indices]  # (1 + negative) * vector_size
  b = model.syn1b[r_indices]  # 1 * (1 + negative)

  # f: outputs of the output-layer nodes
  f = 1. / (1. + exp(- dot(h, r_vecs.T) - b))  # 1 * (1 + negative)

  # gradient of likelihood wrt f times learning rate
  try:
    g = (model.neg_labels[:len(r_indices)] - f) * alpha  # 1 * (1 + negative)
  except ValueError as e:
    print 'negative', model.negative
    print 'f', f.shape
    print 'r_indices', r_indices
    print 'alpha', alpha
    print 'model.neg_labels', model.neg_labels
    raise e

  # gradient of likelihood wrt h times learning rate
  neu1e = dot(g, r_vecs)  # 1 * vector_size

  # learn hidden -> output weights
  model.syn1r[r_indices] += outer(g, h)

  # learn hidden -> output bias
  model.syn1b[r_indices] += g

  # gradient of likelihood wrt c and l times learning rate
  if model.additive:
    neu1e_c = neu1e
    neu1e_l = neu1e
  else:
    # multiplicative
    neu1e_c = np_mult(neu1e, l)
    neu1e_l = np_mult(neu1e, c)

  # learn input vectors and positional matrices
  model.syn0i[i_idxs] += np_mult(hmati, neu1e_c)
  model.syn0k[k_idxs] += np_mult(hmatk, neu1e_c)
  model.hmati += np_mult(i_vecs, neu1e_c)
  model.hmatk += np_mult(k_vecs, neu1e_c)
  model.syn0l[l_idxs] += neu1e_l / len(l_idxs)

def sample_children(model, i_idxs, k_idxs, l_idxs, r_idxs):
  """
  Returns the index of a sampled children based on equation (2) of the
  original paper. The index is in the scope of the given r_idxs list.

  The first half is pretty much the same as train_a_pair().
  """
  assert len(i_idxs) == model.hmati.shape[0]
  assert len(k_idxs) == model.hmatk.shape[0]
  assert len(l_idxs) > 0
  assert len(r_idxs) > 0

  if len(r_idxs) == 1:
    # No choice:
    return 0, (0,)
  else:
    hmati = model.hmati  # memsize_i * vector_size
    hmatk = model.hmatk  # memsize_k * vector_size
    i_vecs = model.syn0i[i_idxs]  # memsize_i * vector_size
    k_vecs = model.syn0k[k_idxs]  # memsize_k * vector_size
    l_vecs = model.syn0l[l_idxs]  # arbitrary * vector_size
    r_vecs = model.syn1r[r_idxs]  # num_candidates * vector_size
    b = model.syn1b[r_idxs]  # 1 * num_candidates

    c = np_mult(i_vecs, hmati).sum(0) + \
        np_mult(k_vecs, hmatk).sum(0)  # 1 * vector_size
    l = l_vecs.mean(0)  # 1 * vector_size

    if model.additive:
      h = c + l  # 1 * vector_size
    else:
      h = np_mult(c, l)

    # weights: outputs of the output-layer nodes
    weights = exp(dot(h, r_vecs.T) + b)  # 1 * num_candidates

    totals = cumsum(weights)
    domain = totals[-1]
    throw = model.random.rand() * domain
    return totals.searchsorted(throw), weights
    # return argmax(weights), weights

def score_ptree(model, i_idxs, k_idxs, l_idxs, r_idxs, r_selected_idx):
  """
  Returns the probability (0-1) of selecting r_selected_idx among r_idxs given
  the input conditions.

  r_selected_idx must be one of r_idxs.
  """
  assert len(i_idxs) == model.hmati.shape[0]
  assert len(k_idxs) == model.hmatk.shape[0]
  assert len(l_idxs) > 0
  assert len(r_idxs) > 0
  assert r_selected_idx in r_idxs

  hmati = model.hmati  # memsize_i * vector_size
  hmatk = model.hmatk  # memsize_k * vector_size
  i_vecs = model.syn0i[i_idxs]  # memsize_i * vector_size
  k_vecs = model.syn0k[k_idxs]  # memsize_k * vector_size
  l_vecs = model.syn0l[l_idxs]  # arbitrary * vector_size
  r_vecs = model.syn1r[r_idxs]  # num_candidates * vector_size
  b = model.syn1b[r_idxs]  # 1 * num_candidates

  c = np_mult(i_vecs, hmati).sum(0) + \
      np_mult(k_vecs, hmatk).sum(0)  # 1 * vector_size
  l = l_vecs.mean(0)  # 1 * vector_size

  if model.additive:
    h = c + l  # 1 * vector_size
  else:
    h = np_mult(c, l)

  # weights: outputs of the output-layer nodes
  weights = exp(dot(h, r_vecs.T) + b)  # 1 * num_candidates

  relative_index = r_idxs.index(r_selected_idx)
  assert relative_index >= 0
  assert relative_index < len(r_idxs)
  score = weights[relative_index] / weights.sum()
  assert score >= 0.
  assert score <= 1.
  return score

class Vocab(object):
  """
  A single vocabulary item, used internally for collecting per-word
  frequency/sampling info. Used for both NL words and production words.
  """
  def __init__(self, **kwargs):
    self.count = 0
    self.__dict__.update(kwargs)

  def __lt__(self, other):  # used for sorting in a priority queue
    return self.count < other.count

  def __str__(self):
    vals = ['%s:%r' % (key, self.__dict__[key])
            for key in sorted(self.__dict__)
            if not key.startswith('_')]
    return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class BiModal(gsutils.SaveLoad):
  def __init__(
      self, train_pairs=None, size=20, alpha=0.05, min_count=5,
      max_vocab_size=None, sample=1e-3, seed=1, workers=4, min_alpha=0.0001,
      negative=5, iter_=5, null_word=True,
      sorted_vocab=1, batch_ptrees=MAX_PTREES_IN_BATCH,
      additive=True, memsize_k=10, memsize_i=10, train_on_init=True):
    """Suffixes:
    _i: internal node in AST
    _k: terminal node in AST
    _l: natural-language word node
    _r: production node
    """
    self.vocab_i = {}  # mapping from a word (string) to a Vocab object
    self.index2word_i = []  # map from a word's matrix index (int) to word (string)
    self.vocab_k = {}
    self.index2word_k = []
    self.vocab_l = {}
    self.index2word_l = []
    self.cum_table = None  # for negative sampling
    self.vector_size = int(size)
    self.vector_size = int(size)
    if size % 4 != 0:
      logger.warning("consider setting layer size to a multiple of 4 for greater performance")
    self.alpha = float(alpha)
    self.max_vocab_size = max_vocab_size
    self.sample = sample
    self.seed = seed
    self.random = random.RandomState(seed)
    self.min_count = min_count
    self.workers = int(workers)
    self.min_alpha = float(min_alpha)
    self.negative = int(negative)
    self.iter = iter_
    self.null_word = null_word
    self.sorted_vocab = sorted_vocab
    self.batch_ptrees = batch_ptrees
    self.additive = additive
    self.memsize_k = memsize_k
    self.memsize_i = memsize_i

    self.train_count = 0
    self.total_train_time = 0

    self.build_vocab(train_pairs)
    if train_on_init:
      self.train(train_pairs)

  def make_cum_table(self, power=0.75, domain=2**31 - 1):
    """
    Create multiple cumulative-distribution tables using stored vocabulary
    word counts for drawing random words in the negative-sampling training
    routines.

    NOTE: this is done only for the production vocab (i.e., vocab_r).

    To draw a word index, choose a random integer up to the maximum value in the
    table (cum_table[-1]), then finding that integer's sorted insertion point
    (as if by bisect_left or ndarray.searchsorted()). That insertion point is the
    drawn index, coming up in proportion equal to the increment at that slot.

    Called internally from 'build_vocab()'.
    """
    self.cum_table = {}
    for parent in self.vocab_r:
      index2word = self.index2word_r[parent]
      vocab_size = len(index2word)
      if vocab_size <= self.negative + NEG_SAMPLING_VOCAB_SIZE_THRESHOLD:
        # For super small vocabs, we just use all negative samples.
        continue
      vocab = self.vocab_r[parent]

      self.cum_table[parent] = zeros(vocab_size, dtype=uint32)
      cum_table = self.cum_table[parent]

      # compute sum of all power (Z in paper)
      train_words_pow = float(sum([vocab[word].count**power for word in vocab]))

      cumulative = 0.0
      for word_index in xrange(vocab_size):
        cumulative += vocab[index2word[word_index]].count**power / train_words_pow
        cum_table[word_index] = round(cumulative * domain)

      if len(cum_table) > 0:
        assert cum_table[-1] == domain

  def build_vocab(self, train_pairs):
    """
    Build vocabulary from a sequence of train_pairs (can be a once-only
    generator stream).

    Each train pair must be a tuple of NL utterance and MyAST object.
    """
    self.scan_vocab(train_pairs)  # initial survey
    self.scale_vocab()
    self.finalize_vocab()  # build tables & arrays

  def scan_vocab(self, train_pairs, progress_per=100):
    """Do an initial scan of all words appearing in sentences."""
    logger.info("collecting all words and their counts")
    train_pair_no = -1

    total_ptrees = 0
    min_reduce = 1

    vocab_l_raw = defaultdict(int)
    vocab_i = set()
    vocab_k = set()
    vocab_r = defaultdict(lambda: defaultdict(int))

    for train_pair_no, (utter, myast) in enumerate(train_pairs):
      if train_pair_no % progress_per == 0:
        logger.info("PROGRESS: at train_pair #%i. Processed %i partial trees.",
                    train_pair_no, total_ptrees)

      l_tokens = utter.split()
      for w in l_tokens:
        vocab_l_raw[w] += 1
      if self.max_vocab_size and len(vocab_l_raw) > self.max_vocab_size:
        # Pruning only applies to NL queries
        gsutils.prune_vocab(vocab_l_raw, min_reduce)
        min_reduce += 1

      for ptree in myast.randomized_partial_trees(
            self.random, self.memsize_k, self.memsize_i):
        terminals, _, parent, children = ptree
        terminals = map(lambda x: x.getSimple(), terminals)
        parent = parent.getSimple()
        if isinstance(children, (list, tuple)):
          children = tuple(map(lambda x: x.getSimple(), children))
        else:
          children = children.getSimple()

        vocab_k |= set(terminals)
        vocab_i.add(parent)
        vocab_r[parent][children] += 1

        total_ptrees += 1

    self.corpus_count = train_pair_no + 1  # this is needed somewhere else

    self.raw_vocab_l = vocab_l_raw

    self.index2word_i = sorted(vocab_i)
    self.index2word_i.insert(0, None)  # 0th position is for the padding token
    self.vocab_i = dict((y,x) for (x,y) in enumerate(self.index2word_i))

    self.index2word_k = sorted(vocab_k)
    self.index2word_k.insert(0, None)  # 0th position is for the padding token
    self.vocab_k = dict((y,x) for (x,y) in enumerate(self.index2word_k))

    self.vocab_r = {}
    self.index2word_r = {}
    self.r_idx_offset = {}
    self.total_r_entries = 0
    for parent in sorted(vocab_r.keys()):
      self.r_idx_offset[parent] = self.total_r_entries
      self.vocab_r[parent] = {}
      self.index2word_r[parent] = []

      # vocab_r[parent] may contain a mixture of SimpleAstNodes and tuples. To
      # sort them properly, we need to sort them separately.
      single_childrens = sorted(x for x in vocab_r[parent].keys()
                                if isinstance(x, SimpleAstNode))
      tuple_childrens = sorted(x for x in vocab_r[parent].keys()
                               if isinstance(x, tuple))
      combined_sorted_children = single_childrens + tuple_childrens
      assert len(combined_sorted_children) == len(vocab_r[parent])
      for children in combined_sorted_children:
        self.vocab_r[parent][children] = Vocab(
          count=vocab_r[parent][children],
          index=len(self.index2word_r[parent]))
        self.index2word_r[parent].append(children)
      self.total_r_entries += len(self.index2word_r[parent])

    """
    After this point, vocab_i and vocab_k are complete.

    The only things left are:

    - scale and finalize vocab_r (because of min_count, downsampling, sorting,
      etc.).
    - build a cum_table for negative sampling of vocab_r.
    """

    logger.info("Collected %i NL word types", len(vocab_l_raw))
    logger.info("Collected %i terminal types", len(self.vocab_k))
    logger.info("Collected %i non-terminal types", len(self.vocab_i))
    logger.info("Created %i production tables", len(self.vocab_r))
    logger.info("Collected %i total production entries", self.total_r_entries)

  def scale_vocab(self):
    """
    Apply vocabulary settings for `min_count` (discarding less-frequent words)
    and `sample` (controlling the downsampling of more-frequent words).

    Delete the raw vocabulary after the scaling is done to free up RAM.
    """
    min_count = self.min_count
    sample = self.sample

    self.index2word_l = []
    self.vocab_l = {}

    drop_unique, drop_total, retain_total, original_total = 0, 0, 0, 0
    retain_words = []
    for word, v in iteritems(self.raw_vocab_l):
      if keep_vocab_item(word, v, min_count):
        retain_words.append(word)
        retain_total += v
        original_total += v
        self.vocab_l[word] = Vocab(count=v, index=len(self.index2word_l))
        self.index2word_l.append(word)
      else:
        drop_unique += 1
        drop_total += v
        original_total += v
    logger.info("min_count=%s retains %s unique NL words (drops %s)",
          min_count, len(retain_words), drop_unique)
    logger.info("min_count leaves a %i-word NL corpus (%i%% of original %i)",
          retain_total, retain_total * 100 / max(original_total, 1), original_total)

    # Precalculate each vocabulary item's threshold for sampling
    if not sample:
      # no words downsampled
      threshold_count = retain_total
    elif sample < 1.0:
      # traditional meaning: set parameter as proportion of total
      threshold_count = sample * retain_total
    else:
      # new shorthand: sample >= 1 means downsample all words with higher count than sample
      threshold_count = int(sample * (3 + sqrt(5)) / 2)

    downsample_total, downsample_unique = 0, 0
    for w in retain_words:
      v = self.raw_vocab_l[w]
      word_probability = (sqrt(v / threshold_count) + 1) * (threshold_count / v)
      if word_probability < 1.0:
        downsample_unique += 1
        downsample_total += word_probability * v
      else:
        word_probability = 1.0
        downsample_total += v
      self.vocab_l[w].sample_int = int(round(word_probability * 2**32))

    logger.info("deleting the raw counts dictionary of %i items", len(self.raw_vocab_l))
    self.raw_vocab_l = defaultdict(int)

    logger.info("sample=%s downsamples %s most-common words", sample, downsample_unique)
    logger.info("downsampling leaves estimated %i word corpus (%.1f%% of prior %i)",
          downsample_total, downsample_total * 100.0 / max(retain_total, 1), retain_total)

    # return from each step: words-affected, resulting-corpus-size
    report_values = {'drop_unique': drop_unique,
                     'retain_total': retain_total,
                     'downsample_unique': downsample_unique,
                     'downsample_total': int(downsample_total)}

    return report_values

  def finalize_vocab(self):
    """Build tables and model weights based on final NL vocabulary settings."""
    if not self.index2word_l:
      self.scale_vocab()
    if self.sorted_vocab:
      self.sort_vocab()

    # build the table for drawing random words (for negative sampling)
    self.make_cum_table()

    if self.null_word:
      # create null pseudo-word for padding when using concatenative L1 (run-of-words)
      # this word is only ever input - never predicted - so count, huffman-point, etc doesn't matter
      word, v = '\0', Vocab(count=1, sample_int=0)
      v.index = len(self.vocab_l)
      self.index2word_l.append(word)
      self.vocab_l[word] = v

    # set initial input/projection and hidden weights
    self.reset_weights()

  def sort_vocab(self):
    """Sort the NL vocabulary so the most frequent words have the lowest indexes."""
    if hasattr(self, 'syn0'):
      raise RuntimeError("must sort before initializing vectors/weights")

    self.index2word_l.sort(key=lambda word: self.vocab_l[word].count, reverse=True)

    for i, word in enumerate(self.index2word_l):
      self.vocab_l[word].index = i

  def _do_train_job(self, train_pairs, alpha, inits):
    """
    Train a single batch of train_pairs. Return 2-tuple `(count of ptrees that
    are actually trained, total raw ptrees)`.
    """
    work, neu1 = inits
    tally = 0
    tally += train_batch(self, train_pairs, alpha, work)
    return tally, self._raw_ptree_count(train_pairs)

  def _raw_ptree_count(self, job):
    """
    Returns the number of ptrees in job.

    job is a sequence of train_pairs.
    """
    num_ptrees = 0
    for utter, myast in job:
      num_ptrees += myast.getPtreeCount()
    return num_ptrees


  def train(self, train_pairs, total_ptrees=None, ptree_count=0,
            total_examples=None, queue_factor=2, report_delay=1.0):
    """
    An example is a pair of a complete AST tree and an utterance.
    A ptree is a partial tree.
    """
    if FAST_VERSION < 0:
      import warnings
      warnings.warn("C extension not loaded; training will be slow.")
      # precompute negative labels optimization for pure-python training
      self.neg_labels = zeros(self.negative + 1 + NEG_SAMPLING_VOCAB_SIZE_THRESHOLD)
      self.neg_labels[0] = 1.

    logger.info(
        "training model with %i workers",
        self.workers)

    if not self.vocab_i or not self.vocab_k or not self.vocab_l or not self.vocab_r:
      raise RuntimeError("you must first build vocabulary before training the model")
    if not hasattr(self, 'syn0i') or not hasattr(self, 'syn0k') or not hasattr(self, 'syn0l'):
      raise RuntimeError("you must first finalize vocabulary before training the model")

    if total_ptrees is None and total_examples is None:
      if self.corpus_count:
        total_examples = self.corpus_count
        logger.info("expecting %i train pairs, matching count from corpus used for vocabulary survey", total_examples)
      else:
        raise ValueError("you must provide either total_ptrees or total_examples, to enable alpha and progress calculations")

    job_tally = 0

    if self.iter > 1:
      train_pairs = gsutils.RepeatCorpusNTimes(train_pairs, self.iter)
      total_ptrees = total_ptrees and total_ptrees * self.iter
      total_examples = total_examples and total_examples * self.iter

    def worker_loop():
      """Train the model, lifting lists of train_pairs from the job_queue."""

      # per-thread private work memory - useless in numpy implementation
      work = matutils.zeros_aligned(self.vector_size, dtype=REAL)
      neu1 = matutils.zeros_aligned(self.vector_size, dtype=REAL)
      jobs_processed = 0
      while True:
        job = job_queue.get()
        if job is None:
          progress_queue.put(None)
          break  # no more jobs => quit this worker
        train_pairs, alpha = job
        tally, raw_tally = self._do_train_job(train_pairs, alpha, (work, neu1))
        progress_queue.put((len(train_pairs), tally, raw_tally))  # report back progress
        jobs_processed += 1
      logger.debug("worker exiting, processed %i jobs", jobs_processed)

    def job_producer():
      """Fill jobs queue using the input `train_pairs` iterator."""
      job_batch, batch_size = [], 0
      pushed_ptrees, pushed_examples = 0, 0
      next_alpha = self.alpha
      job_no = 0

      for train_pair in train_pairs:
        train_pair_length = self._raw_ptree_count([train_pair])

        # can we fit this train_pair into the existing job batch?
        if batch_size + train_pair_length <= self.batch_ptrees:
          # yes => add it to the current job
          job_batch.append(train_pair)
          batch_size += train_pair_length
        else:
          # no => submit the existing job
          logger.debug(
            "queueing job #%i (%i ptrees, %i train_pairs) at alpha %.05f",
            job_no, batch_size, len(job_batch), next_alpha)
          job_no += 1
          job_queue.put((job_batch, next_alpha))

          # update the learning rate for the next job
          if self.min_alpha < next_alpha:
            if total_examples:
              # examples-based decay
              pushed_examples += len(job_batch)
              progress = 1.0 * pushed_examples / total_examples
            else:
              # ptrees-based decay
              pushed_ptrees += self._raw_ptree_count(job_batch)
              progress = 1.0 * pushed_ptrees / total_ptrees
            next_alpha = self.alpha - (self.alpha - self.min_alpha) * progress
            next_alpha = max(self.min_alpha, next_alpha)

          # add the train_pair that didn't fit as the first item of a new job
          job_batch, batch_size = [train_pair], train_pair_length

      # add the last job too (may be significantly smaller than batch_ptrees)
      if job_batch:
        logger.debug(
          "queueing job #%i (%i ptrees, %i train_pairs) at alpha %.05f",
          job_no, batch_size, len(job_batch), next_alpha)
        job_no += 1
        job_queue.put((job_batch, next_alpha))

      if job_no == 0 and self.train_count == 0:
        logger.warning(
          "train() called with an empty iterator (if not intended, "
          "be sure to provide a corpus that offers restartable "
          "iteration = an iterable)."
        )

      # give the workers heads up that they can finish -- no more work!
      for _ in xrange(self.workers):
        job_queue.put(None)
      logger.debug("job loop exiting, total %i jobs", job_no)

    # buffer ahead only a limited number of jobs.. this is the reason we can't
    # simply use ThreadPool :(
    job_queue = Queue(maxsize=queue_factor * self.workers)
    progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

    workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
    unfinished_worker_count = len(workers)
    workers.append(threading.Thread(target=job_producer))

    for thread in workers:
      thread.daemon = True  # make interrupting the process with ctrl+c easier
      thread.start()

    example_count, trained_ptree_count, raw_ptree_count = 0, 0, ptree_count
    start, next_report = default_timer() - 0.00001, 1.0

    while unfinished_worker_count > 0:
      report = progress_queue.get()  # blocks if workers too slow
      if report is None:  # a thread reporting that it finished
        unfinished_worker_count -= 1
        logger.info("worker thread finished; awaiting finish of %i more threads", unfinished_worker_count)
        continue
      examples, trained_ptrees, raw_ptrees = report
      job_tally += 1

      # update progress stats
      example_count += examples
      trained_ptree_count += trained_ptrees  # only ptrees in vocab & sampled
      raw_ptree_count += raw_ptrees

      # log progress once every report_delay seconds
      elapsed = default_timer() - start
      if elapsed >= next_report:
        if total_examples:
          # examples-based progress %
          logger.info(
            "PROGRESS: at %.2f%% examples, %.0f ptrees/s, in_qsize %i, out_qsize %i",
            100.0 * example_count / total_examples, trained_ptree_count / elapsed,
            gsutils.qsize(job_queue), gsutils.qsize(progress_queue))
        else:
          # ptrees-based progress %
          logger.info(
            "PROGRESS: at %.2f%% ptrees, %.0f ptrees/s, in_qsize %i, out_qsize %i",
            100.0 * raw_ptree_count / total_ptrees, trained_ptree_count / elapsed,
            gsutils.qsize(job_queue), gsutils.qsize(progress_queue))
        next_report = elapsed + report_delay

    # all done; report the final stats
    elapsed = default_timer() - start
    logger.info(
      "training on %i raw ptrees (%i effective ptrees) took %.1fs, %.0f effective ptrees/s",
      raw_ptree_count, trained_ptree_count, elapsed, trained_ptree_count / elapsed)
    if job_tally < 10 * self.workers:
      logger.warn("under 10 jobs per worker: consider setting a smaller `batch_ptrees' for smoother alpha decay")

    # check that the input corpus hasn't changed during iteration
    if total_examples and total_examples != example_count:
      logger.warn("supplied example count (%i) did not equal expected count (%i)", example_count, total_examples)
    if total_ptrees and total_ptrees != raw_ptree_count:
      logger.warn("supplied raw word count (%i) did not equal expected count (%i)", raw_ptree_count, total_ptrees)

    self.train_count += 1  # number of times train() has been called
    self.total_train_time += elapsed
    return trained_ptree_count

  def reset_weights(self):
    """Reset all projection weights to an initial (untrained) state, but keep
    the existing vocabulary."""
    logger.info("resetting layer weights")
    self.syn0i = empty((len(self.vocab_i), self.vector_size), dtype=REAL)
    self.syn0k = empty((len(self.vocab_k), self.vector_size), dtype=REAL)
    self.syn0l = empty((len(self.vocab_l), self.vector_size), dtype=REAL)
    self.hmati = ones((self.memsize_i, self.vector_size), dtype=REAL) / self.memsize_i
    self.hmatk = ones((self.memsize_k, self.vector_size), dtype=REAL) / self.memsize_k
    self.syn1r = empty((self.total_r_entries, self.vector_size), dtype=REAL)
    self.syn1b = empty(self.total_r_entries, dtype=REAL)

    # randomize weights vector by vector, rather than materializing a huge
    # random matrix in RAM at once
    for i in xrange(len(self.vocab_i)):
      # construct deterministic seed from word AND seed argument
      self.syn0i[i] = self.seeded_vector(str(self.index2word_i[i]) + str(self.seed))
    for i in xrange(len(self.vocab_k)):
      self.syn0k[i] = self.seeded_vector(str(self.index2word_k[i]) + str(self.seed))
    for i in xrange(len(self.vocab_l)):
      self.syn0l[i] = self.seeded_vector(str(self.index2word_l[i]) + str(self.seed))
    for i in xrange(self.total_r_entries):
      self.syn1r[i] = self.seeded_vector('syn1r' + str(i) + str(self.seed))
    for parent in self.index2word_r:
      offset = self.r_idx_offset[parent]
      sum_count = sum(self.vocab_r[parent][w].count for w in self.index2word_r[parent])
      for j in xrange(len(self.index2word_r[parent])):
        # log(P(w|parent))
        self.syn1b[offset+j] = log(self.vocab_r[parent][self.index2word_r[parent][j]].count / sum_count)

  def seeded_vector(self, seed_string, vector_size=None):
    """Create one 'random' vector (but deterministic by seed_string)"""
    if not vector_size:
      vector_size = self.vector_size
    # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
    once = random.RandomState(hash(seed_string) & 0xffffffff)
    return (once.rand(vector_size) - 0.5) / vector_size

  def getSampleCallback(self, utter):
    """
    Returns a callback function to be used by MyAST._sample().

    utter is an NL utterance, in the format of a string.
    """
    l_idxs = get_l_idxs(self, utter)

    def _callback(terminals, ancestors, parent):
      if not l_idxs: return ()
      terminals_simple = map(lambda x:x.getSimple(), terminals)
      k_idxs = [self.vocab_k[x] for x in reversed(terminals_simple) if x in self.vocab_k]
      k_idxs += [0] * (self.memsize_k - len(k_idxs))  # padding

      ancestors_simple = map(lambda x:x.getSimple(), ancestors)
      i_idxs = [self.vocab_i[x] for x in ancestors_simple if x in self.vocab_i]
      i_idxs += [0] * (self.memsize_i - len(i_idxs))  # padding

      parent = parent.getSimple()
      r_idxs = find_r_idxs_sample(self, parent)

      # weights are for debugging output only
      ret = sample_children(self, i_idxs, k_idxs, l_idxs, r_idxs)
      r_sampled_idx, weights = ret
      children_sampled = self.index2word_r[parent][r_sampled_idx]


      if len(self.index2word_r[parent]) > 1:
        logger.debug('')
        logger.debug('Parent: %s'%parent)
        for i in range(len(self.index2word_r[parent])):
          selected = '  <--' if i == r_sampled_idx else ''
          logger.debug('    %s - %s%s'%(
              weights[i], self.index2word_r[parent][i], selected))
        logger.debug('Terminals: %s'%terminals_simple)
        logger.debug('Ancestors: %s'%ancestors_simple)
        logger.debug('')
      else:
        logger.debug('%s -> %s'%(parent, children_sampled))

      return children_sampled

    return _callback

  def scoreFullTree(self, utter, myast_node):
    """
    Returns a calculated score between 0 and 1:
      2 * sigmoid of mean of log-likelihood of all valid ptrees.

    A valid ptree is a tree that:
     - the expected children tuple is in vocabulary
    """
    l_idxs = get_l_idxs(self, utter)
    if not l_idxs:
      raise ValueError('No valid NL word. Should turn on null_word option in training.')

    sum_logp = 0.
    count_valid = 0
    for ptree in myast_node.partial_trees(self.memsize_k, self.memsize_i):
      terminals, ancestors, parent, children = ptree

      terminals_simple = map(lambda x:x.getSimple(), terminals)
      # setting unrecognized terminal to the padding symbol
      k_idxs = [self.vocab_k[x] if x in self.vocab_k else 0 for x in reversed(terminals_simple)]
      k_idxs += [0] * (self.memsize_k - len(k_idxs))  # padding

      ancestors_simple = map(lambda x:x.getSimple(), ancestors)
      # setting unrecognized non-terminal to the padding symbol
      i_idxs = [self.vocab_i[x] if x in self.vocab_i else 0 for x in ancestors_simple]
      i_idxs += [0] * (self.memsize_i - len(i_idxs))  # padding

      parent = parent.getSimple()
      # absolute indexes of all possible children tuples under this parent
      r_idxs = find_r_idxs_sample(self, parent)

      if len(r_idxs) == 1:
        # no need to predict at all
        continue

      if isinstance(children, (list, tuple)):
        children = tuple(map(lambda x: x.getSimple(), children))
      else:
        children = children.getSimple()

      if not parent in self.vocab_r or \
         not children in self.vocab_r[parent]:
        # then it is not a valid partial tree
        continue

      r_selected_idx = self.vocab_r[parent][children].index + self.r_idx_offset[parent]
      p = score_ptree(self, i_idxs, k_idxs, l_idxs, r_idxs, r_selected_idx)
      sum_logp += log(p)
      count_valid += 1

    if count_valid == 0:
      return 0
    else:
      # 2. on numerator will scale (0, 0.5) to (0, 1)
      return 2. / (1. + exp(- sum_logp / count_valid))


  def __str__(self):
    return "%s(vocab_i=%s, vocab_k=%s, vocab_l=%s, vocab_r=%s parents, " \
           "vector_size=%s, alpha=%s)" % (
      self.__class__.__name__, len(self.index2word_i), len(self.index2word_k),
      len(self.index2word_l), len(self.index2word_r), self.vector_size, self.alpha)

  def save(self, *args, **kwargs):
    # don't bother storing the cum_table, which is recalculable
    kwargs['ignore'] = kwargs.get('ignore', ['cum_table'])
    super(BiModal, self).save(*args, **kwargs)

  save.__doc__ = gsutils.SaveLoad.save.__doc__

  @classmethod
  def load(cls, *args, **kwargs):
    model = super(BiModal, cls).load(*args, **kwargs)
    if hasattr(model, 'index2word_l'):
      model.make_cum_table()  # rebuild cum_table (which is not saved) from vocabulary

    if not hasattr(model, 'random'):
      model.random = random.RandomState(model.seed)

    if not hasattr(model, 'train_count'):
      model.train_count = 0
      model.total_train_time = 0

    return model

if __name__ == '__main__':
  MEMSIZE = 10

  logging.basicConfig(
      format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
      # level=logging.INFO)
      level=logging.DEBUG)
  logging.info("running %s", " ".join(sys.argv))
  logging.info("using optimization %s", FAST_VERSION)

  code = """plt.plot([1,2,3], linestyle='--')
plt.title('Hello World')"""
  node = ast.parse(code)
  myast = MyAST(node=node)
  utter = "plot a dashed line and set title to ' hello world '"
  train_pairs = [(utter, myast)]

  model = BiModal(train_pairs, size=20, min_count=None, workers=1, iter_=100,
                  sample=None,  # don't do downsampling
                  additive=True, memsize_k=MEMSIZE, memsize_i=MEMSIZE,
                  # Caution:
                  #  - if alpha is too big: will trigger parameter explosion and
                  #    numerical overflow
                  #  - if alpha is too small: will take forever to converge.
                  alpha=0.25,
                  seed=random.randint(0, 2**16-1),
                 )

  # testing save and reload
  fname = 'output/tmp-test-saved.model'
  model.save(fname)
  model = BiModal.load(fname)

  my_callback = model.getSampleCallback(utter)
  myast_sampled = MyAST.sample_from_root(
        memsize_k=MEMSIZE, memsize_i=MEMSIZE, callback=my_callback)

  code_sampled = astunparse.unparse(myast_sampled.node)
  print 'Reconstructed Code:'
  print code_sampled

  score = model.scoreFullTree(utter, myast_sampled)
  print 'Score:', score
