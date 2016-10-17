"""
Utility classes of element transformation and lookup

"""
import csv

from codemend import relative_path
from codemend.models.annotate_code_with_api import get_fu_fau

class Element:
  def __init__(self, elem, node, val_node=None, parent_node=None):
    """
    Parameters
    ----------
    - var_node: for kwarg and dict-key elem only.

    """
    self.elem = elem
    self.node = node
    self.val_node = val_node
    self.parent_node = parent_node
    self.lineno = self.get_lineno()

  def get_lineno(self):
    if hasattr(self.node, 'lineno'):
      return self.node.lineno
    elif self.parent_node and hasattr(self.parent_node, 'lineno'):
      return self.parent_node.lineno
    else:
      return -1

  def getTextRange(self):
    """
    Returns (lineno, col_offset, end_lineno, end_col_offset)

    """
    if self.val_node:
      return self.node.lineno, self.node.col_offset, \
             self.val_node.end_lineno, self.val_node.end_col_offset
    else:
      return self.node.lineno, self.node.col_offset, \
             self.node.end_lineno, self.node.end_col_offset

  def getValueTextRange(self):
    """
    Returns (lineno, col_offset, end_lineno, end_col_offset)

    """
    if self.val_node:
      return self.val_node.lineno, self.val_node.col_offset, \
             self.val_node.end_lineno, self.val_node.end_col_offset

  def __repr__(self):
    return self.elem

def unsimplify(elem_id):
  """
  Hard-coded reverse-mapping of stype_map.

  """
  fields = elem_id.split('@')
  full_name = elem_id
  if fields[0].startswith('plt.gca.'):
    full_name = elem_id.replace('plt.gca.', 'matplotlib.axes.Axes.')
  elif fields[0].startswith('plt.'):
    full_name = elem_id.replace('plt.', 'matplotlib.pyplot.')
  elif fields[0].startswith('ax.'):
    full_name = elem_id.replace('ax.', 'matplotlib.axes.Axes.')
  elif fields[0].startswith('mpl.'):
    full_name = elem_id.replace('mpl.', 'matplotlib.')
  return full_name

def get_pyplot_funcs():
  # used by other part as well
  with open(relative_path('lib/matplotlib/pyplot.py')) as reader:
    pyplot_src = reader.read()
  pyplot_funcs = []
  for line in pyplot_src.split('\n'):
    line = line.strip()
    if line.startswith('def '):
      line = line[len('def '):]
      fields = line.split('(')
      func_name = fields[0]
      pyplot_funcs.append(func_name)
  return pyplot_funcs

pyplot_funcs = set(get_pyplot_funcs())

def transform_pylab_elem(elem_str):
  """
  pylab.xxx --> pyplot.xxx  (if exists)

  """
  if not elem_str: return elem_str
  fields = elem_str.split('.')
  if fields[0] == 'pylab':
    if len(fields) > 1:
      f = fields[1].split('@', 1)[0]
      if f in pyplot_funcs:
        fields[0] = 'matplotlib.pyplot'
  return '.'.join(fields)

def transform_pltgca_elem(elem_str):
  """
  plt.gca.xxx --> plt.xxx (if exists)
  plt.gcf.xxx --> plt.xxx (if exists)
  plt.figure.xxx --> plt.xxx (if exists)

  NOTE: elem_str should already be simplified.

  """
  if not elem_str: return elem_str
  fields = elem_str.split('.')
  if len(fields) >= 3 and \
     fields[0] == 'plt' and \
     fields[1] in ('gca', 'gcf', 'figure'):
    f = fields[2].split('@', 1)[0]
    if f in pyplot_funcs:
      fields.pop(1)
    elif f.startswith('set_') and f[len('set_'):] in pyplot_funcs:
      fields[2] = f[len('set_'):]
      fields.pop(1)
  return '.'.join(fields)


class ElementNormalizer:
  """
  Handles static transformation of elements.

  Performs the following two types of transformations:
  1. rule-based simplication (e.g., matplotlib.pyplot -> plt)

  2. return type inference
   - NOTE:
     - return type inference should be disabled when semantic information is
       required, e.g., in plt.gca@spines@set_visible, we want to keep
       "spines", because it tells us what do we want set_visible for.
     - however, for documentation lookup (and thus training data generation),
       return type inference is very important.

  The following are not accepted, and should be handled by alias resolution
  mechanism prior to using the functions of this class:
   - aliases (i.e., module aliases created by import-as expressions)
   - local variables assigned via function calls

  To edit the stype_map and rtype_map rules, go to:
    https://docs.google.com/spreadsheets/d/1irlsls3kO6YliL2BU7kwPqSIB8EAdq5GRdZpovNS6Iw/edit#gid=0
  and save the spreadsheets as rtype_map.csv and stype_map.csv in this folder.

  """

  def __init__(self):
    # load simplification mapping
    self.stype_map = {}
    with open(relative_path('docstring_parse/annotation/stype_map.csv'), 'rb') as csvfile:
      reader = csv.reader(csvfile)
      for fields in reader:
        assert len(fields) == 2
        assert fields[0]
        assert fields[1]
        self.stype_map[fields[0]] = fields[1]

    # load rtype mapping
    self.rtype_map = {}
    with open(relative_path('docstring_parse/annotation/rtype_map.csv'), 'r') as csvfile:
      reader = csv.reader(csvfile)
      for fields in reader:
        assert len(fields) == 2
        assert fields[0]
        assert fields[1]
        fields = map(self.simplify, fields)
        self.rtype_map[fields[0]] = fields[1]

    self.fu, self.fau = get_fu_fau()

  def simplify(self, elem):
    """
    Apply core_simplify (based on human annotation in stype_map.csv) plus some
    additional heuristic transformations.

    """
    assert isinstance(elem, basestring), type(elem)
    # the order here of these additional cleanings is important
    elem = transform_pylab_elem(elem)
    elem = self.core_simplify(elem)
    elem = transform_pltgca_elem(elem)
    return elem

  def unsimplify(self, elem):
    return unsimplify(elem)

  def core_simplify(self, elem):
    """
    Try to simplify the longest possible prefix.

    Returns the simplified element.

    Note: at most one rule will fire. No recursive applications of rules.

    """
    assert isinstance(elem, basestring), elem.__class__.__name__
    fields = elem.split('.')
    while fields:
      test_str = '.'.join(fields)
      if test_str in self.stype_map:
        return elem.replace(test_str, self.stype_map[test_str], 1)
      fields.pop()
    return elem

  def infer(self, elem, simplify_result=True):
    """
    Try to infer the actual type of the element, but will ignore the last part
    of the element.

    Examples:
    - plt.gca -> plt.gca
    - plt.gca.set_xticklabels -> ax.set_xticklabels
    - plt.legend.XXX -> mpl.legend.Legend

    NOTE: in the above example, "ax" is not a variable name, but the
    simplified form of matplotlib.axes.Axes.

    Returns the inferred element.

    Note: will try to recursively match return rules.

    """
    assert isinstance(elem, basestring), elem.__class__.__name__

    elem = self.simplify(elem)
    fields = elem.split('@', 1)
    fields2 = fields[0].split('.')
    if len(fields2) > 1:
      head_elem = '.'.join(fields2[:-1])
      head_elem = self.core_infer(head_elem)
      elem = '.'.join((head_elem, fields2[-1]))
      if len(fields) > 1: elem += '@' + fields[1]

    if simplify_result:
      return self.simplify(elem)
    else:
      return self.unsimplify(elem)

  def core_infer(self, elem):
    """
    Try to iteratively infer the actual type of the given string by matching
    the longest possible key (simplified elem_id) in rtype_map. Operating on
    simplified elements.

    Examples:
    - plt.gca -> ax
    - plt.legend -> mpl.legend.Legend

    """
    assert isinstance(elem, basestring)

    fields = elem.split('.')
    while fields:
      test_str = '.'.join(fields)
      if test_str in self.rtype_map:
        new_elem = elem.replace(test_str, self.rtype_map[test_str], 1)
        if new_elem == elem: return elem
        else: return self.core_infer(new_elem)
      else:
        fields.pop()
    return elem

if __name__ == '__main__':
  enorm = ElementNormalizer()

  print
  print 'Test starts...'
  assert enorm.simplify('matplotlib.pyplot') == 'plt'
  assert enorm.simplify('matplotlib.pyplot.grid') == 'plt.grid'
  assert enorm.simplify('matplotlib.pyplot.gca.bar') == 'plt.bar'
  assert enorm.simplify('matplotlib.pyplot.gca.set_title') == 'plt.title'
  assert enorm.simplify('matplotlib.pyplot.gcf.suptitle') == 'plt.suptitle'
  assert enorm.simplify('matplotlib.pyplot.figure.legend') == 'plt.legend'

  # Not ax.title. Because we applied aggressive simplification rules to reduce
  # sparsity.
  assert enorm.infer('plt.gca.set_title') == 'plt.title'
  assert enorm.infer('plt.gca.set_title', False) == 'matplotlib.pyplot.title'
  assert enorm.infer('plt.gca.set_xticklabels', False) == 'matplotlib.axes.Axes.set_xticklabels'
  assert enorm.infer('plt.legend@loc') == 'plt.legend@loc'
  assert enorm.infer('plt.legend.XXX@YYY') == 'mpl.legend.Legend.XXX@YYY'

  print 'All assertion tests passed.'
  print

  print 'Below are human-inspection tests:'
  print enorm.doc('matplotlib.pyplot.plot@color')
  print enorm.doc('matplotlib.pyplot.title@bbox@pad')
  print
