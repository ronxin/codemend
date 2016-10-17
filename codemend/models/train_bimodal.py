"""
Train bimodal model.
"""
import argparse
import ast
import logging
import numpy as np
import os
import sys

import bimodal
import myast

from backup_util import BackupHandler


if __name__ == '__main__':
  # check and process cmdline input
  program = os.path.basename(sys.argv[0])
  if len(sys.argv) < 2:
      print(globals()['__doc__'] % locals())
      sys.exit(1)

  np.seterr(all='raise')  # don't ignore numpy errors

  parser = argparse.ArgumentParser()
  parser.add_argument("-backup_folder", default="output/backup")
  parser.add_argument("-train_pairs", help="pickle of training data", required=True)
  parser.add_argument("-save_model", help="output file name", required=True)
  parser.add_argument("-size", help="Size of vectors", type=int, default=20)
  parser.add_argument("-alpha", help="initial learning rate", type=float, default=0.05)
  parser.add_argument("-min_count", type=int, default=5)
  parser.add_argument("-max_vocab_size", default=None)
  parser.add_argument("-sample", help="down sampling factor", type=float, default=1e-3)
  parser.add_argument("-seed", type=int, default=1)
  parser.add_argument("-workers", type=int, default=1)
  parser.add_argument("-min_alpha", type=float, default=0.0001)
  parser.add_argument("-negative", help="number of negative samples", type=int, default=5)
  parser.add_argument("-iter", help="number of epochs to train", type=int, default=5)
  parser.add_argument("-additive", help="additive or muliplicative", type=bool, default=True)
  parser.add_argument("-memsize_k", type=int, default=10)
  parser.add_argument("-memsize_i", type=int, default=10)
  parser.add_argument("-verbose", type=str, default="INFO")
  parser.add_argument("-sample_train_pairs", type=int, default=0)

  args = parser.parse_args()

  verbose_level = logging.DEBUG if args.verbose.upper() == 'DEBUG' else logging.INFO
  logging.basicConfig(
      format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
      level=verbose_level)

  bh = BackupHandler(args.backup_folder)

  train_pairs = bh.load(args.train_pairs)

  if args.sample_train_pairs > 0 and len(train_pairs) > args.sample_train_pairs:
    # This is for debugging purpose
    import random
    train_pairs = random.sample(train_pairs, args.sample_train_pairs)

  logging.info("Generating MyAST instances.")

  train_pairs = [
      (utter.decode('utf-8'),
       myast.MyAST(node=ast.parse(code)))
      for utter, code in train_pairs]

  model = bimodal.BiModal(
      train_pairs = train_pairs,
      size = args.size,
      alpha = args.alpha,
      min_count = args.min_count,
      max_vocab_size = args.max_vocab_size,
      sample = args.sample,
      seed = args.seed,
      workers = args.workers,
      min_alpha = args.min_alpha,
      negative = args.negative,
      iter_ = args.iter,
      additive = args.additive,
      memsize_k = args.memsize_k,
      memsize_i = args.memsize_i,
      train_on_init=True)

  logging.info("Saving Model")

  file_name = args.save_model
  model.save(file_name)

  logging.info("Done.")
