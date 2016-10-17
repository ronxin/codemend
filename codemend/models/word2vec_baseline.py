"""
A baseline method for highlighting functions and parameters based on word2vec
model.

Two variations:
 1. Use the function / argument name as the representation of itself.
 2. Use the centroid of the API documentations as the representation of a
    function / argument.
"""

from __future__ import division
from collections import defaultdict
from gensim import matutils
from gensim.models.word2vec import Word2Vec
from numpy import float32 as REAL
from word2vec_util import load_gensim_from_binary_file
import numpy as np
import pattern.en
import pattern.vector

from baseline import Baseline
from ngram_util import ngram_partition
from annotate_code_with_api import extractCallComponents

def tokenize(s):
  return ' '.join(pattern.en.tokenize(s)).split()

def stem(token):
  return pattern.vector.stem(token, stemmer=pattern.vector.PORTER)

class Word2vecBaseline(Baseline):
  def __init__(self, w2v_model, maxngram=1, fu_fau=None, name=None, use_stem=False):
    """
    w2v_model can be a binary vectors file, or a loaded gensim model instance.

    If fu_fau is not None, then use variation 2. Otherwise variation 1.

    fu_fau = (fu, fau)
      - fu: [func_id] = utter
      - fau: [func_id, arg] = utter
    """
    self.maxngram = maxngram
    self.name = name
    self.use_stem = use_stem

    if isinstance(w2v_model, basestring):
      # it is a file name
      # self.model = Word2Vec.load_word2vec_format(w2v_model, binary=True)
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

    self.bow = False
    if fu_fau:
      self.bow = True  # use bag-of-word representation for funcs and args

      print 'Word2vecBaseline building function/argument indexes...'

      assert isinstance(fu_fau, tuple)
      assert len(fu_fau) == 2
      fu, fau = fu_fau

      self.idfs = self.get_idf(fu.values() + fau.values())

      self.func_ids = sorted(fu.keys())
      self.func_lookup = dict((y,x) for (x,y) in enumerate(self.func_ids))
      f_vecs = []
      for f in self.func_ids:
        u = fu[f]
        v = self.get_bow_representation(u + ' ' + f)
        f_vecs.append(v)
      self.f_vecmat = np.array(f_vecs)
      assert self.f_vecmat.shape == (len(self.func_ids), self.model.vector_size)

      self.func_args = defaultdict(list)
      for f,a in fau:
        self.func_args[f].append(a)

      self.func_args = dict(self.func_args)
      self.a_vecmat = {}  # [func_id] = 2d-array
      self.func_arg_lookup = {}  # [func_id][arg] = idx
      for f in self.func_args:
        self.func_args[f] = sorted(self.func_args[f])
        self.func_arg_lookup[f] = dict((y,x) for (x,y) in enumerate(self.func_args[f]))
        a_vecs = []
        for a in self.func_args[f]:
          u = fau[f, a]
          u_func = fu[f] if f in fu else ''
          v = self.get_bow_representation(' '.join((u, a, u_func, f)))
          a_vecs.append(v)
        self.a_vecmat[f] = np.array(a_vecs)
        assert self.a_vecmat[f].shape == (
          len(self.func_args[f]), self.model.vector_size)

      print 'Finished building indexes.'

      # At this point, the following variables are initialized:
      # self.
      #   f_vecmat, a_vecmat
      #   func_ids, func_args
      #   func_lookup, func_arg_lookup

  def get_word_indexes(self, query):
    """
    Input will be tokenized and matched against the given vocabulary.

    If `maxngram` > 1, then n-gram partition is performed prior to matching.

    TODO: considers removing stop words. The `ngram_partition` function
    already supports this.
    """
    tokens = tokenize(query)
    if self.use_stem:
      tokens = map(stem, tokens)
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

  def rank_funcs(self, query, funcs, parent):
    """
    The score of each function is the maximum of the following:
     - similarity(func, query)
     - similarity(func + arg, query) for any arg in the current call

    If self.bow == True, then will try to use parent.call_nodes to get
    functions and arguments and funcs will be ignored. If parent.call_nodes is
    not available, then funcs will be used.

    TODO: make use of the value of arguments in searching, too. e.g., "red" as
          in color="red".

    """
    query = query.lower()
    self.model.init_sims()
    q_vec = self.get_bow_representation(query)
    if self.bow \
        and parent is not None \
        and hasattr(parent, 'call_nodes'):
      funcs = []
      scores = []
      for call in parent.call_nodes:
        func, keywords = extractCallComponents(call)
        tmp_vecmat = np.zeros((1 + len(keywords), self.model.vector_size))
        score = 0
        if func in self.func_lookup:
          func_idx = self.func_lookup[func]
          tmp_vecmat[0] += self.f_vecmat[func_idx]
          if func in self.func_arg_lookup:
            for i,arg in enumerate(keywords):
              if arg in self.func_arg_lookup[func]:
                arg_idx = self.func_arg_lookup[func][arg]
                tmp_vecmat[i+1] += self.a_vecmat[func][arg_idx]
          tmp_scores = np.dot(q_vec, tmp_vecmat.T)
          score = tmp_scores.max()
        funcs.append(func)
        scores.append(score)
    elif self.bow:
      func_vecmat = np.zeros((len(funcs), self.model.vector_size))
      for i, func in enumerate(funcs):
        func_idx = self.func_lookup[func]
        func_vecmat[i] += self.f_vecmat[func_idx]
      scores = np.dot(q_vec, func_vecmat.T)
    else:
      func_vecmat = np.zeros((len(funcs), self.model.vector_size))
      for i, func in enumerate(funcs):
        if func in self.model.vocab:
          func_idx = self.model.vocab[func].index
          func_vecmat[i] += self.model.syn0norm[func_idx]
      scores = np.dot(q_vec, func_vecmat.T)

    sorted_funcs = sorted(zip(funcs, scores), key=lambda x:x[1], reverse=True)

    return sorted_funcs

  def rank_args(self, query, func, args, parent=None):
    query = query.lower()
    self.model.init_sims()
    q_vec = self.get_bow_representation(query)
    arg_vecmat = np.zeros((len(args), self.model.vector_size))

    for i, arg in enumerate(args):
      if self.bow:
        assert func in self.func_arg_lookup, func
        if arg in self.func_arg_lookup[func]:
          arg_idx = self.func_arg_lookup[func][arg]
          arg_vecmat[i] += self.a_vecmat[func][arg_idx]
      else:
        if arg in self.model.vocab:
          arg_idx = self.model.vocab[arg].index
          arg_vecmat[i] += self.model.syn0norm[arg_idx]
    scores = np.dot(q_vec, arg_vecmat.T)
    sorted_args = sorted(zip(args, scores), key=lambda x:x[1], reverse=True)
    return sorted_args

  def __repr__(self):
    suffix = '_func_as_seq' if self.bow else '_func_as_term'
    return self.name + suffix

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
  import ast
  from annotate_code_with_api import findCallNodes

  class Test:
    def __init__(self):
      TEST_VECTOR_BIN_FILE = 'output/vectors-flat-mpl-0205.bin'
      # TEST_VECTOR_BIN_FILE = 'output/vectors-so-text-python-5gram.bin'
      # TEST_VECTOR_BIN_FILE = 'output/vectors-so-text-python-stem-3gram.bin'
      MAXNGRAM = 3

      for i in range(2):
        print '\nRound %d\n'%i

        if i == 0:
          print 'Loading a big vector file. Will take a while....'
          wb = Word2vecBaseline(TEST_VECTOR_BIN_FILE,
                                maxngram=MAXNGRAM)
        else:
          from annotate_code_with_api import get_fu_fau
          fu_fau = get_fu_fau()
          wb = Word2vecBaseline(wb.model,
                                maxngram=MAXNGRAM,
                                fu_fau=fu_fau)

        TEST_CODE = """plt.bar(x, y, color="red")
plt.title('hello world')
plt.xlim(1,6)"""
        node = ast.parse(TEST_CODE)
        self.call_nodes = findCallNodes(node)
        funcs = [extractCallComponents(x)[0] for x in self.call_nodes]
        results = wb.rank_funcs('set colors of the faces', funcs, self)
        for x in results:
          print x[0], x[1]

        print '-------------'

        results = wb.rank_args(
          'add shadow to legend',
          'legend',
          ['shadow', 'bbox_to_anchor', 'fontsize'])
        for x in results:
          print x[0], x[1]

  Test()
