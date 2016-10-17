# -*- coding: utf-8 -*-
import ast

# http://astunparse.readthedocs.org/en/latest/installation.html
import astunparse

def is_ascii(mystring):
  try:
    mystring.encode('ascii')
  except UnicodeEncodeError:
    return False
  except UnicodeDecodeError:
    return False
  return True

class MyAST:
  """
  Wrapper around AST nodes. Supports parent finding.

  Functionalities of MyAST:
   [1] take in AST tree root, find all partial trees
   [2] for each partial tree, provide
       - last K terminal nodes (DPS) (with padding)
       - last I non-terminal nodes (going to parent)
       - context non-terminal node
       - target children tuple
   [3] support the sampling of a tree out of the model
       - take record of generated children tuples
       - continuously yields the next set of context:
         - last K terminal nodes
         - last I non-terminal ndoe
         - context non-terminal node
       - convert the generated tree back to python's generic AST tree
       - unparse the python's AST tree and generate actual python code
       - test the above pipeline by going from code -> tree -> code

  Not using ast.NodeVisitor or ast.NodeTransformer because we need to apply
  complex mutation on the nodes.
  """
  def __init__(self, parent=None, node=None, fv_pair=None, field_name=None,
               content=None):
    self.parent = parent
    if node:
      self.node = node
      self.children = []
      for name, value in ast.iter_fields(node):
        if value is None: continue
        self.children.append(MyAST(parent=self, fv_pair=(name, value)))
    elif fv_pair:
      field, value = fv_pair
      self.field = field
      if isinstance(value, ast.AST):
        self.children = MyAST(parent=self, node=value)
      elif isinstance(value, (list, tuple)):
        self.children = [MyAST(parent=self, node=x) for x in value]
      else:
        assert value is not None
        self.children = MyAST(parent=self, content=value)
    elif field_name:
      # Used only for sampling AST based on a trained model. During sampling
      # at certain time points, only the field_name is known, and the children
      # are unknown.
      self.field = field_name
    elif content is not None:
      self.content = content
    else:
      raise ValueError('Either node, fv_pair, field_name, or content should be' \
                       'specified.')

    if self.parent is None:
      self.num_ptrees = self._visit()

  def _visit(self):
    """
    Initial depth-first visit, for serializing terminals.

    Returns the number of non-terminal nodes (including self).
    """
    if self.parent:
      self.terminals = self.parent.terminals
    else:
      self.terminals = []
    self.offset = len(self.terminals)  # record the current number of visited terminals

    num_ptrees = 0
    if hasattr(self, 'children'):
      num_ptrees += 1
      if isinstance(self.children, (list, tuple)):
        for child in self.children:
          num_ptrees += child._visit()
      else:
        num_ptrees += self.children._visit()
    else:
      assert hasattr(self, 'content')
      self.terminals.append(self)
    return num_ptrees

  def partial_trees(self, memsize_k, memsize_i):
    """Returns a generator of partial trees, each of which is represented by a
    tuple that contains:
     - a list of at most memsize_k terminal nodes visited prior to this node
     - a list of at most memsize_i ancestors: parent, grandparent, ...
     - the current node
     - current node's children

     Note: partially trees are generated only for non-terminal nodes.
    """
    if hasattr(self, 'children'):
      terminals = self.terminals[max(0, self.offset-memsize_k):self.offset]
      ancestors = self._find_ancestors(memsize_i)

      yield terminals, ancestors, self, self.children

      children = self.children
      if not isinstance(children, (list, tuple)):
        # It is necessary this way to distinguish a list of one element vs.
        # a non-list variable
        children = [children]
      for child in children:
        for ptree in child.partial_trees(memsize_k, memsize_i):
          yield ptree

  def randomized_partial_trees(self, random, memsize_k, memsize_i):
    ptrees = [x for x in self.partial_trees(memsize_k, memsize_i)]
    random.shuffle(ptrees)
    return ptrees

  def _find_ancestors(self, max_returned=None):
    ancestors = []
    cur_node = self.parent
    while cur_node:
      if max_returned and len(ancestors) == max_returned:
        break
      ancestors.append(cur_node)
      cur_node = cur_node.parent
    return ancestors

  def __repr__(self):
    if hasattr(self, 'node'):
      return 'NODE_'+self.node.__class__.__name__
    elif hasattr(self, 'field'):
      return 'FIELD_'+self.field
    else:
      assert hasattr(self, 'content')
      return unicode(self.content)

  @staticmethod
  def sample_from_root(memsize_k, memsize_i, callback):
    root = MyAST(parent=None, node=ast.Module())
    root._sample(memsize_k, memsize_i, callback)
    return root

  def _sample(self, memsize_k, memsize_i, callback):
    """
    Uses the callback function to iteratively construct a tree, guaranteeing
    the integrity of the actual AST.

    callback takes three parameters: (terminals, ancestors, parent)
    callback returns: children = [SimpleAstNode] OR SimpleAstNode
    """
    if self.parent:
      self.terminals = self.parent.terminals
    else:
      self.terminals = []
    self.offset = len(self.terminals)
    terminals = self.terminals[max(0, self.offset-memsize_k):self.offset]
    ancestors = self._find_ancestors(memsize_i)
    sampled_children = callback(terminals, ancestors, self)
    if sampled_children:
      if isinstance(sampled_children, (list, tuple)):
        first_child = sampled_children[0]
      else:
        first_child = sampled_children
      if hasattr(first_child, 'node_type'):
        assert hasattr(self, 'field')
        assert hasattr(self.parent, 'node')
        if isinstance(sampled_children, (list, tuple)):
          nodes = [x.node_type() for x in sampled_children]
          setattr(self.parent.node, self.field, nodes)
          self.children = [MyAST(parent=self, node=x) for x in nodes]
          for child in self.children:
            child._sample(memsize_k, memsize_i, callback)
        else:
          node = first_child.node_type()
          setattr(self.parent.node, self.field, node)
          self.children = MyAST(parent=self, node=node)
          self.children._sample(memsize_k, memsize_i, callback)
      elif hasattr(first_child, 'field_name'):
        assert hasattr(self, 'node')
        for child in sampled_children: assert hasattr(child, 'field_name')
        self.children = [MyAST(parent=self, field_name=x.field_name)
                         for x in sampled_children]
        # Not necessary to set the actual field values of self.node. They will
        # be set by the children nodes.
        for child in self.children:
          child._sample(memsize_k, memsize_i, callback)
      else:
        assert not isinstance(sampled_children, (list, tuple))
        assert hasattr(sampled_children, 'content')
        assert hasattr(self, 'field')
        assert hasattr(self.parent, 'node')
        content = sampled_children.content
        setattr(self.parent.node, self.field, content)
        self.children = MyAST(parent=self, content=content)
        self.terminals.append(self.children)

    # Fix missing fields
    if hasattr(self, 'node'):
      for field in self.node._fields:
        if not hasattr(self.node, field):
          if field in ('keywords', 'decorator_list', 'args', 'defaults'):
            setattr(self.node, field, [])
          else:
            setattr(self.node, field, None)

  def getSimple(self):
    """Returns a SimpleAstNode representation of self."""
    if hasattr(self, 'node'):
      return SimpleAstNode(node_type=self.node.__class__)
    elif hasattr(self, 'field'):
      return SimpleAstNode(field_name=self.field)
    else:
      assert hasattr(self, 'content')
      return SimpleAstNode(content=self.content)

  def getPtreeCount(self):
    if hasattr(self, 'num_ptrees'):
      return self.num_ptrees
    else:
      raise ValueError('num_ptrees not calculated for sampled trees.')

class SimpleAstNode:
  """Represents a simple AST node without parent/child links."""
  def __init__(self, node_type=None, field_name=None, content=None):
    if node_type is not None:
      self.node_type = node_type
      self._tuple = ('NODE', node_type, None, None)
    elif field_name is not None:
      self.field_name = field_name
      self._tuple = ('FIELD', None, field_name, None)
    elif content is not None:
      if isinstance(content, basestring) and not is_ascii(content):
        content = 'UNK'  # unknown character
      self.content = content
      self._tuple = ('CONTENT', None, None, content)
    else:
      raise ValueError('Either node_type, or field_name, or content must be specified.')

  def __hash__(self):
    return hash(self._tuple)

  def __eq__(self, other):
    return self._tuple == other._tuple

  def __lt__(self, other):
    try:
      return self._tuple < other._tuple
    except TypeError:
      # e.g., the content may be a complex number
      return str(self._tuple) < str(other._tuple)

  def __repr__(self):
    if hasattr(self, 'node_type'):
      return 'NODE_'+self.node_type.__name__
    elif hasattr(self, 'field_name'):
      return 'FIELD_'+self.field_name
    else:
      assert hasattr(self, 'content')
      return str(self.content)

if __name__ == '__main__':
  """
  Test:
  1. generate partial trees for a given AST
  2. build mapping from partial tree to children tuple
  3. sample a tree using learned mapping
  4. convert the sampled tree to code
  """
  code = """plt.plot(a,
    [1,2,3],
    linestyle='--')
plt.title('北京欢迎你')
def f(x): print 'abc'
for i in range(10): x = i
import c as d
from e import f
pass
class A(B):
  def __init__(self):
    pass
# Some comments (will be lost)
f = lambda x: x**2"""
  node = ast.parse(code)
  myast = MyAST(node=node)

  mapping = {}  # [terminals, ancestors, parent] = children
  for ptree in myast.partial_trees(5, 5):
    terminals, ancestors, parent, children = ptree
    terminals = map(lambda x: x.getSimple(), terminals)
    ancestors = map(lambda x: x.getSimple(), ancestors)
    parent = parent.getSimple()
    if isinstance(children, (list, tuple)):
      children = map(lambda x: x.getSimple(), children)
    else:
      children = children.getSimple()
    key = tuple(terminals), tuple(ancestors), parent
    mapping[key] = children

  def my_callback(terminals, ancestors, parent):
    terminals = map(lambda x: x.getSimple(), terminals)
    ancestors = map(lambda x: x.getSimple(), ancestors)
    parent = parent.getSimple()
    return mapping[tuple(terminals), tuple(ancestors), parent]

  myast_sampled = MyAST.sample_from_root(5, 5, my_callback)

  code_sampled = astunparse.unparse(myast_sampled.node)
  print 'Reconstructed Code:'
  print code_sampled
  print 'Number of ptrees:', myast.getPtreeCount()
