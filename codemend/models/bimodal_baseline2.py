from gensim.models import Word2Vec

from codemend.models.word2vec_util import load_gensim_from_binary_file
from codemend.models.baseline2 import Baseline, SuggestItem
from codemend.models.bimodal2 import BiModal
from codemend.models.constraint import prune

class BiModalBaseline(Baseline):
   def __init__(self, name, model_file, w2v_model):
     self.name = name

     self.model = BiModal.load(model_file)

     if isinstance(w2v_model, basestring):
       w2v_model = load_gensim_from_binary_file(w2v_model)
     else:
       assert isinstance(w2v_model, Word2Vec)
     self.model.w2v_model = w2v_model

     self.model.syn0l = self.model.w2v_model.syn0

   def suggest(self, query, context):
     used_elems = context.used_elems()
     scores = self.model.score_all(query, used_elems)
     elems_sorted = sorted(zip(scores, self.model.all_elems), reverse=True)
     suggest_sorted = [SuggestItem(elem=elem, score=score) for (score, elem) in elems_sorted]
     suggest_pruned = prune(used_elems, suggest_sorted)
     return suggest_pruned[:50]

   def __repr__(self):
     return self.name

if __name__ == '__main__':
  from codemend import relative_path

  bmb = BiModalBaseline('tmp',
      relative_path('models/output/bi2-test.model'),
      relative_path('models/output/vectors-flat-mpl-0205.bin')
    )
  print bmb, 'initialized.'
