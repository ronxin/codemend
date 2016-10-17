"""
A baseline method that supports global leaf-path searching.

A lot of stuff is taken from word2vec_baseline.py

"""

from __future__ import division
from gensim import matutils
from gensim.models.word2vec import Word2Vec
from numpy import float32 as REAL
from word2vec_util import load_gensim_from_binary_file
import math
import numpy as np
import pattern.en
import pattern.vector
import re

from codemend import relative_path, BackupHandler
from codemend.docstring_parse import doc_serve

from baseline2 import Baseline, SuggestItem
from ngram_util import ngram_partition
from annotate_code_with_api import get_fu_fau
from element import ElementNormalizer

def tokenize(s):
  return ' '.join(pattern.en.tokenize(s)).split()

def lemma(token):
  return pattern.vector.stem(token, stemmer=pattern.vector.LEMMA)

class Word2vecBaseline(Baseline):
  def __init__(self, w2v_model, all_elem_counts, maxngram=1,
               name=None, use_lemma=True,
               heuristic=False, use_coke=False):
    """
    w2v_model can be a binary vectors file, or a loaded gensim model instance.

    """
    self.maxngram = maxngram
    self.name = name
    self.use_lemma = use_lemma
    assert isinstance(all_elem_counts, dict)
    self.all_elem_counts = all_elem_counts
    self.heuristic = heuristic
    self.use_coke = use_coke

    if isinstance(w2v_model, basestring):
      self.model = load_gensim_from_binary_file(w2v_model)
      self.model.filename = w2v_model.split('/')[-1]
      if not self.name:
        self.name = self.model.filename
    else:
      assert isinstance(w2v_model, Word2Vec)
      self.model = w2v_model
      if not self.name:
        if hasattr(self.model, 'filename'):
          self.name = self.model.filename


    self.model.init_sims()  # normalize the vectors

    self.enormer = ElementNormalizer()

    if self.use_coke:
      bh = BackupHandler(relative_path('models/output/backup'))
      coke_file = 'coke_0329'
      if not bh.exists(coke_file):
        raise ValueError('Coke file does not exist: %s'%coke_file)
      self.coke = bh.load(coke_file)

    print 'Trying to load element indexes from cache ...'
    bh = BackupHandler(relative_path('models/output/backup'))
    elem_index_backup_name = self.model.filename + '_elem_index'
    if bh.exists(elem_index_backup_name):
      self.idfs, self.elems, self.elem_lookup, self.vecmat = bh.load(elem_index_backup_name)

    else:
      print 'Word2vecBaseline building element indexes...'

      fu, fau = get_fu_fau()
      self.idfs = self.get_idf(fu.values() + fau.values())

      self.elems = sorted(self.all_elem_counts.keys())
      self.elem_lookup = dict((y,x) for (x,y) in enumerate(self.elems))
      vecs = []
      for e in self.elems:
        u = doc_serve.get_training_doc(e, True)
        v = self.get_bow_representation(u)
        vecs.append(v)
      self.vecmat = np.array(vecs)
      assert self.vecmat.shape == (len(self.elems), self.model.vector_size)

      bh.save(elem_index_backup_name, (self.idfs, self.elems, self.elem_lookup, self.vecmat))

      print 'Finished building indexes.'

    # At this point, the following variables are initialized:
    # self.
    #   vecmat
    #   elems, elem_lokup

  def get_word_indexes(self, query):
    """
    Input will be tokenized and matched against the given vocabulary.

    If `maxngram` > 1, then n-gram partition is performed prior to matching.

    TODO: considers removing stop words. The `ngram_partition` function
    already supports this.
    """
    tokens = tokenize(query)
    if self.use_lemma:
      tokens = map(lemma, tokens)
    if self.maxngram > 1:
      tokens_ngrams = ngram_partition(' '.join(tokens), self.model.vocab)
      tokens = list(set(tokens) | set(tokens_ngrams))
    idxs = [self.model.vocab[w].index for w in tokens if w in self.model.vocab]
    if not idxs:
      idxs.append(0)  # the null word
    return idxs

  def get_bow_representation(self, query):
    """
    Returns the mean vector.

    Vectors are weighted by inverse document frequency.

    """
    idxs = self.get_word_indexes(query)

    if hasattr(self, 'idfs'):
      idf_weights = [self.idfs[x] for x in idxs]

      # tricky !!!  intention: stop word removal
      idf_weights = map(lambda x: x if x > 2 else 0, idf_weights)
    else:
      assert not self.bow
      idf_weights = [1] * len(idxs)

    idf_weights = np.array(idf_weights).reshape((1,-1))
    raw_vecs = self.model.syn0norm[idxs]
    weighted_sum = np.dot(idf_weights, raw_vecs)

    weighted_average = (weighted_sum / len(idxs))[0]
    return matutils.unitvec(weighted_average).astype(REAL)

  def suggest(self, query, context):
    # TODO: use annoy for nearest neighbor searching.

    query = query.lower()
    self.model.init_sims()
    q_vec = self.get_bow_representation(query)

    scores = np.dot(q_vec, self.vecmat.T)

    used_elems = context.used_elems()
    used_elem_set = set(used_elems)
    if self.heuristic:
      # For each element, assign the following awards:
      # 1: score *= 2.0 - exact match with an used element (skip 2 if matched)
      # 2: score *= 1.5 - its prefix matches an used element
      # 3: score *= log(freq + 10) - smoothed global frequency award
      used_elem_re = re.compile('^' + '|'.join(re.escape(p) for p in used_elems))
      for i, score in enumerate(scores):
        e = self.elems[i]
        if e in used_elem_set:
          scores[i] *= 2
        elif used_elem_re.match(e):
          scores[i] *= 1.5

        scores[i] *= math.log(self.all_elem_counts[e] + 10)

    sorted_elems = sorted(zip(self.elems, scores), key=lambda x:x[1], reverse=True)

    def additional_filter(elem):
      if self.all_elem_counts[elem] < 10: return False
      # Very aggressive. Essentially forcing short-sightedness.
      if '@' in elem and elem.split('@',1)[0] not in used_elem_set: return False
      return True
    sorted_elems = filter(lambda x:additional_filter(x[0]), sorted_elems)
    sorted_elems = sorted_elems[:100]


    out = [SuggestItem(elem=elem,score=score) for (elem,score) in sorted_elems]

    if self.use_coke:
      # Using cooccurrence to rerank the top-ranked items
      coke_score = 0.0
      for item in out:
        elem = item.elem
        for elem_used in used_elems:
          coke_score += float(self.coke_lookup(elem, elem_used)) \
                        / (self.all_elem_counts[elem_used] + 1)
        item.score *= math.log(coke_score + 10)

    out = sorted(out, key=lambda x:x.score, reverse=True)
    return out

  def score(self, query, used_elems, candidate_elems):
    query = query.lower()
    self.model.init_sims()
    q_vec = self.get_bow_representation(query)
    r_elems = filter(lambda x: x in self.elem_lookup, candidate_elems)
    if not r_elems: return []
    r_idxs = map(lambda x: self.elem_lookup[x], r_elems)
    r_vecs = self.vecmat[r_idxs]
    scores = np.dot(q_vec, r_vecs.T)
    return zip(r_elems, scores)

  def coke_lookup(self, x, y):
    if (x,y) in self.coke:
      return self.coke[x,y]
    elif (y,x) in self.coke:
      return self.coke[y,x]
    return 0

  def __repr__(self):
    return self.name

  def get_idf(self, documents):
    """
    Get inverse document frequency based on a given vocabulary and a given
    corpus (`documents`). For words in the vocabulary that are unseen in the
    given corpus, assign an IDF of 5.

    documents: a list of strings.

    Returns a list of docfreqs. Indexed by word index.

    """
    counts = [0] * len(self.model.index2word)
    num_doc = len(documents)
    for doc in documents:
      word_idxs = set(self.get_word_indexes(doc))
      for wi in word_idxs:
        counts[wi] += 1
    idfs = []
    for i in xrange(len(counts)):
      if counts[i] > 0:
        idfs.append(np.log(num_doc / counts[i]))
      else:
        # tricky !!!
        idfs.append(5)
    return idfs


if __name__ == "__main__":
  """
  Testing word2vec baseline

  """
  TEST_VECTOR_BIN_FILE = 'output/vectors-flat-mpl-0205.bin'
  # TEST_VECTOR_BIN_FILE = 'output/vectors-so-text-python-5gram.bin'
  # TEST_VECTOR_BIN_FILE = 'output/vectors-so-text-python-stem-3gram.bin'

  from codemend.models.eval2 import ContextBuilder

  cb = ContextBuilder()

  MAXNGRAM = 1
  wb = Word2vecBaseline(TEST_VECTOR_BIN_FILE, cb.getAllElementCounts(),
                        MAXNGRAM, 'test-w2v', use_lemma=True,
                        heuristic=False, use_coke=True)

  query = 'add legend'
  with open(relative_path('demo/code-samples/before_afters/before1.py')) as reader:
    code = reader.read()
  context = cb.getContext(code)
  results = wb.suggest(query, context)
  for r in results:
    assert isinstance(r, SuggestItem)
    print '%.3f\t%s\t%d'%(r.score, r.elem, wb.all_elem_counts[r.elem])
