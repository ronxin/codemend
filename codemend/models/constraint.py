"""
Apply constraints (filtering) to a list of suggested items.

The contraints are:
- #1: Not suggesting another plotting function or its argv if a plotting function
  has already been called, except the function itself which can be suggested.
- #2: Not suggesting a non-plotting function if there is no plotting function
  called.
- #3: Not recommending elements that occur too infrequently.
- #4: When a function is not used before, and its argv is recommended, we strip
  the "@", and recommend the function first, followed by the argv. e.g.
  [pie@0, pie] => [pie, pie@0].

"""

from codemend import BackupHandler, relative_path
from codemend.demo.code_suggest import get_plot_commands
from codemend.models.baseline2 import SuggestItem

plot_commands = get_plot_commands()
plot_commands_set = set(plot_commands)

bh = BackupHandler(relative_path('experimental/code_suggest/output/backup'))
elem_counts = bh.load('elem_pyplot_counts_0404')

def prune(used_elems, suggest_elems):
  for elem in used_elems:
    assert isinstance(elem, basestring)
  for elem in suggest_elems:
    assert isinstance(elem, SuggestItem), type(elem)

  used_elems_set = set(used_elems)
  used_funcs = map(get_func_name, used_elems)
  used_funcs_set = set(used_funcs)
  has_used_plot_commands = any(map(lambda x: x in plot_commands_set, used_funcs))

  filtered_suggests = []
  for suggest in suggest_elems:
    # Rule #3
    if not suggest.elem in elem_counts: continue
    if elem_counts[suggest.elem] < 10: continue

    # Rules #1, #2
    f = get_func_name(suggest.elem)
    if f in used_funcs_set:
      filtered_suggests.append(suggest)
    elif has_used_plot_commands and f not in plot_commands_set:
      filtered_suggests.append(suggest)
    elif not has_used_plot_commands and f in plot_commands_set:
      filtered_suggests.append(suggest)

  filtered_suggests_2 = []
  seen_func_ids_set = set()
  for suggest in filtered_suggests:
    func_id = get_func_id(suggest.elem)
    # Rule #4
    if func_id not in used_elems_set \
        and func_id not in seen_func_ids_set \
        and '@' in suggest.elem:
      suggest_stripped = SuggestItem(elem=func_id, score=suggest.score)
      seen_func_ids_set.add(func_id)
      filtered_suggests_2.append(suggest_stripped)
    elif func_id not in seen_func_ids_set:
      filtered_suggests_2.append(suggest)

  return filtered_suggests_2


def get_func_name(elem):
  if not elem: return elem
  assert isinstance(elem, basestring)

  func_id = get_func_id(elem)
  return func_id.split('.')[-1]

def get_func_id(elem):
  if not elem: return elem
  assert isinstance(elem, basestring)
  return elem.split('@', 1)[0]

# Testing:
#   All functions here can be tested in test_bimodal2.py
