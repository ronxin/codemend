"""
Train bimodal model.
"""
import ast
import astunparse
import logging
import random
import sys

import bimodal
import myast

from backup_util import BackupHandler


if __name__ == '__main__':
  bh = BackupHandler('output/backup')
  train_pairs = bh.load('train_pairs_0204')

  train_pairs_sample = random.sample(train_pairs, 100)

  train_pairs_sample = [
      (utter.decode('utf-8'),
       myast.MyAST(node=ast.parse(code)))
      for utter, code in train_pairs_sample]

  MEMSIZE = 10

  logging.basicConfig(
      format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
      level=logging.INFO)
      # level=logging.DEBUG)
  logging.info("running %s", " ".join(sys.argv))

  model = bimodal.BiModal(
      train_pairs = train_pairs_sample,
      size = 20,
      min_count = None,
      workers = 12,
      iter_ = 100,
      null_word = True,
      sample = None,
      additive = True,
      memsize_k = MEMSIZE,
      memsize_i = MEMSIZE,
      alpha = 0.05,
      seed = 1,
      train_on_init=True)

  model.save('output/model-100sample-0204')

