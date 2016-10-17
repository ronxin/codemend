from codemend import BackupHandler, relative_path
from codemend.models.element import ElementNormalizer
from codemend.models.word2vec_util import load_gensim_from_binary_file
from codemend.models.bimodal2 import BiModal
from codemend.experimental.code_suggest.mine_element import code_examples

if __name__ == '__main__':
  bh = BackupHandler(relative_path('experimental/code_suggest/output/backup'))
  elem_counts = bh.load('elem_pyplot_counts_0404')
  all_elems = sorted(elem_counts.keys())
  all_elems_counts = elem_counts
  enormer = ElementNormalizer()
  w2v_model = load_gensim_from_binary_file(
    relative_path('models/output/vectors-so-text-python-lemma-win5.bin'))  # <-- note the change here!!

  model = BiModal(all_elems, all_elems_counts, w2v_model, code_examples, enormer,
                  threads=None, alpha=0.05, window=5, negative=20,
                  additive=0, multiply=0, concat=1,
                  epoch=1, rand_parent_doc=True,
                  hint_pvecs_init=True, hint_rvecs_init=False,
                  neg_sample_used_elem=False)

  model.save(relative_path('models/output/bi2-0410-t.model'))

  # Changes:
  # bi2-test -- lastest gold version for user study
  # bi2-0410-a -- epoch=10, fixed stopwords (e.g., excluding bar from stopwords) -- this is vanilla
  # bi2-0410-b -- epoch=1, quick check if setting is all right.
  # bi2-0410-c -- epoch=10, replicating bi2-0410-a
  # bi2-0410-d -- epoch=1, randomly with-parent doc
  # bi2-0410-e -- epoch=5, randomly with-parent doc
  # bi2-0410-f -- epoch=1, vanilla + additive
  # bi2-0410-g -- epoch=1, vanilla + multiply
  # bi2-0410-h -- epoch=1, window=15  (vanilla window was 5) + concat
  # bi2-0410-i -- epoch=1, hint_rvecs_init
  # bi2-0410-j -- epoch=1, neg_sample_used_elem
  # bi2-0410-k -- epoch=1, positional arg doc inference (permanent change)
  # bi2-0410-l -- epoch=1, randomly with parent doc
  # bi2-0410-m -- epoch=1, randomly with parent doc, alpha=0.01 (vanilla alpha=0.05)
  # bi2-0410-n -- epoch=1, randomly with parent doc, alpha=0.1
  # bi2-0410-o -- epoch=5, randomly with parent doc, alpha=0.01
  # bi2-0410-p -- epoch=5, randomly with parent doc, alpha=0.1
  # bi2-0410-q -- epoch=1, randomly with parent doc, alpha=0.25
  # bi2-0410-r -- epoch=1, replicate d (disabled positional arg doc inference, made in k, permanent change)
  # bi2-0410-s -- epoch=1, word2vec window=3
  # bi2-0410-t -- epoch=1, word2vec window=5
