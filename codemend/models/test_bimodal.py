import bimodal
import ast
import astunparse
import myast
import numpy as np
import logging
import sys

if __name__ == '__main__':
  logging.basicConfig(
      format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
      level=logging.INFO)
      # level=logging.DEBUG)
  logging.info("running %s", " ".join(sys.argv))


  model = bimodal.BiModal.load('output/model-0205-k100-iter100_with_args')
  model.random = np.random.RandomState()

  # Test 1: Sampling test
  # Test 2: Ranking test

  # utter = 'create a figure of size 12 inches and 5 inches'
  # utter = 'set the y axis label to bar'
  # utter = 'set the title of the figure to baz'
  # utter = 'bzzzzzzzzzzzzzzzzzzzzz'
  # utter = 'set the face color facecolor to red'
  # utter = 'make it red'
  utter = 'set the face color'

  sample_callback = model.getSampleCallback(utter)
  myast_sampled = myast.MyAST.sample_from_root(
    memsize_k = model.memsize_k,
    memsize_i = model.memsize_i,
    callback = sample_callback)

  code_sampled = astunparse.unparse(myast_sampled.node)
  print
  print
  print 'Reconstructed code'
  print code_sampled.strip()
  print
  print

  code1 = 'plt.gca().set_xlabel("foo")'
  code2 = 'plt.ylabel("bar")'
  code3 = 'plt.title("baz")'
  code4 = 'plt.set_facecolor("red")'
  code5 = 'plt.figure(figsize=(12,5))'
  codes = [code1, code2, code3, code4, code5]
  nodes = [ast.parse(x) for x in codes]
  myast_nodes = [myast.MyAST(node=x) for x in nodes]
  scores = [model.scoreFullTree(utter, x) for x in myast_nodes]
  for c, s in zip(codes, scores):
    print s, c
