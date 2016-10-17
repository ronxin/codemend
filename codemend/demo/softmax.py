"""
Utils for softmax for color highlights.

"""
from bisect import bisect_right
import numpy as np

def softmaxed_weights_for_highlight(weighted_tups):
  if not weighted_tups: return []

  all_scores = sorted(x[1] for x in weighted_tups)
  score_lookup = {}
  for obj, score in weighted_tups:
    invrank = bisect_right(all_scores, score)
    new_score = ((invrank * 2 / len(all_scores)) ** 3) * score
    score_lookup[obj] = new_score

  # softmax
  score_sum = np.sum(np.exp(score_lookup.values()))
  for obj in score_lookup:
    score_lookup[obj] = min(1, np.exp(score_lookup[obj]) / score_sum * 2)

  # fix for the winner group - make the winner cluster (if any)  closer
  max_old_score = all_scores[-1]
  max_new_score = max(score_lookup.values())
  old_score_lookup = dict(weighted_tups)
  for obj in score_lookup:
    diff = max_old_score - old_score_lookup[obj]
    if diff < 0.1:
      score_lookup[obj] = max(max_new_score - diff * 0.5, 0)
    elif diff < 0.2:
      score_lookup[obj] = max(max_new_score - diff * 2.0, 0)
    elif diff < 0.3:
      score_lookup[obj] = max(max_new_score - diff * 3.0, 0)

  # new scores
  new_weighted_tups = [(obj, score_lookup[obj]) for (obj, _) in weighted_tups]
  return new_weighted_tups


if __name__ == '__main__':
  print 'Test softmaxed_weights_for_highlight ...'

  print 'Co-winning'
  print softmaxed_weights_for_highlight([
    ('a',0.8),('b',0.8),('c',0.5),('d',0.2)])

  print 'Co-medium-winning'
  print softmaxed_weights_for_highlight([
    ('a',0.6),('b',0.6),('c',0.4),('d',0.4)])

  print 'Clear winner'
  print softmaxed_weights_for_highlight([
    ('a',0.9),('b',0.2),('c',0.1),('d',0.05)])

  print 'Very close winners'
  print softmaxed_weights_for_highlight([
    ('a',0.9),('b',0.899),('c',0.1),('d',0.05),('e',0.002)])

  print 'Close winners'
  print softmaxed_weights_for_highlight([
    ('a',0.9),('b',0.85),('c',0.1),('d',0.05),('e',0.002)])

  print 'Close medium winners'
  print softmaxed_weights_for_highlight([
    ('a',0.6),('b',0.57),('c',0.3),('d',0.2),('e',0.002)])

  print 'Many co-winners'
  print softmaxed_weights_for_highlight([
    ('a',0.9),('b',0.89),('c',0.88),('d',0.87),('e',0.002)])

