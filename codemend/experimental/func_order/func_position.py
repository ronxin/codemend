"""
Given code and functions, automatically find the best place to insert a
function.

Algorithm:
- Load average position per function computed offline from ~5K code examples.
- Given a code snippet and a function:
  - extract call nodes in the code
  - score all possible positions (represented as the line number to insert
    before). When scoring, compare the target function with all other
    functions. If the average position is in favored order, add positive
    reward to the score; otherwise add negative penalty.
  - return all scored positions.

Complexity:
 - Per serving, O(N * 2). N: number of matplotlib-API functions called in the
   given code snippet.

Limitation:
 - No consideration of indentation or control structures.

"""
import ast

from codemend import BackupHandler, relative_path
from codemend.models.annotate_code_with_api import findCallNodes, extractCallComponents

class FuncPositionFinder:
  def __init__(self):
    bh = BackupHandler(relative_path('demo/data'))
    self.pos_ave = bh.load('pos_ave')
    print 'FuncPositionFinder: loaded %d average positions for functions'%len(self.pos_ave)

  def findPositions(self, code, func):
    """
    Returns a list of tuples: [(Position, Score)].

    Position is the line number before which the new function call is to be
    inserted. If the line number doesn't exist in the current code, then it
    means the new function call should be appended at the end.

    Note: the line numbers start at 1 (not 0).

    """
    # TODO: change parsing to jedi
    try:
      node = ast.parse(code)
    except SyntaxError:
      return []
    maxlineno = len(code.split('\n'))

    calls = findCallNodes(node)
    called_funcs = [extractCallComponents(x)[0] for x in calls]
    filterd_funcs_calls = filter(lambda (x,y): x in self.pos_ave, zip(called_funcs, calls))
    linenos = map(lambda (x,y): y.lineno, filterd_funcs_calls)  # candiate line numbers
    cand_linenos = linenos + [maxlineno]

    if not func in self.pos_ave:
      return [(l,0) for l in cand_linenos]
    my_pos_ave = self.pos_ave[func]

    out = []
    for cl in cand_linenos:
      score = 0.
      for l, (f, _) in zip(linenos, filterd_funcs_calls):
        assert f in self.pos_ave
        pos_diff = my_pos_ave - self.pos_ave[f]
        # note: "cl=l" means "insert before"
        if (pos_diff > 0 and cl > l) or (pos_diff < 0 and cl <= l):
          score += abs(pos_diff)
        else:
          score -= abs(pos_diff)
      out.append((cl, score))
    return out


if __name__ == '__main__':
  from textwrap import dedent

  position_finder = FuncPositionFinder()
  code = dedent('''
  abc = np.random.randn(100)
  plt.plot(abc)
  plt.xlabel("hello world")
  ''')
  test_funcs = ['title', 'xkcd', 'subplots', 'show']
  for f in test_funcs:
    print f
    print position_finder.findPositions(code, f)
    print
