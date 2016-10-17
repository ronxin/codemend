import sys
sys.path.append('../thonny')

import ast
import csv
import code as code_module
import re

import ast_utils
import mpl_handler

class CodeAnalyzer:

  def __init__(self):
    self.func_ids = set()
    with open('../docstring_parse/fu.csv', 'rb') as csvfile:
      reader = csv.reader(csvfile)
      next(reader, None)
      for f,_ in reader:
        self.func_ids.add(f)

  class ASTVisitor(ast.NodeVisitor):
    def __init__(self):
      self.funcs = []

    def visit_Call(self, node):
      if hasattr(node, 'func'):
        func = node.func
        assert hasattr(node, 'keywords')
        self.funcs.append((func.lineno - 1, func.end_lineno,
          func.col_offset, func.end_col_offset, node.keywords))

    def get_funcs(self):
      return self.funcs

  def findFunctionsFromCodeBlock(self, codeBlock):
    for lineGroup_offset, lineGroup in self.generateLineGroups(codeBlock):
      code_unit = '\n'.join(lineGroup)
      node = ast.parse(code_unit)
      ast_utils.mark_text_ranges(node, unicode(code_unit))
      v = self.ASTVisitor()
      v.visit(node)
      funcs = v.get_funcs()
      exec(code_unit)
      for line_offset, end_line_offset, col_offset, end_col_offset, keywords in funcs:
        func_str =  self.getFuncStr(lineGroup, line_offset, end_line_offset, col_offset, end_col_offset)
        func_id = self.getFuncID(func_str, locals())
        yield func_id, \
          (line_offset, col_offset), \
          (end_line_offset, end_col_offset), keywords

  def getFuncStr(self, lineGroup, line_offset, end_line_offset, col_offset, end_col_offset):
    if line_offset == end_line_offset - 1:
      return lineGroup[line_offset][col_offset:end_col_offset]
    elif line_offset < end_line_offset - 1:
      return '\n'.join(
          [lineGroup[line_offset][col_offset:]] +
          lineGroup[line_offset+1:end_line_offset-1] +
          [lineGroup[end_line_offset-1][:end_col_offset]]
        )
    else:
      raise ValueError('Invalid line_offset=%d, end_line_offset=%d'%(line_offset, end_line_offset))

  def generateLineGroups(self, codeBlock):
    lines = codeBlock.split('\n')
    lines = map(lambda x:x.rstrip(), lines)
    begin_lineno = 0
    end_lineno = 1
    while end_lineno <= len(lines):
      addLineBreak = False
      if end_lineno == len(lines): addLineBreak = True
      elif lines[end_lineno] and lines[end_lineno][0] not in (' ', '\t'): addLineBreak = True

      cur_code_unit = '\n'.join(lines[begin_lineno:end_lineno])

      cur_code_unit = re.sub(r'\n+', '\n', cur_code_unit)
      cur_code_unit = cur_code_unit.rstrip()

      if cur_code_unit:
        if addLineBreak: cur_code_unit += '\n'

        if self.isCodeComplete(cur_code_unit):
          yield begin_lineno, [''] * begin_lineno + lines[begin_lineno:end_lineno]
          begin_lineno = end_lineno
      else:
        begin_lineno += 1

      end_lineno += 1

  def isCodeComplete(self, code):
    ret = code_module.compile_command(code)
    return ret is not None

  def getFuncID(self, func_str, globals_):
    func_obj = eval(func_str, globals_)
    func_tokens = func_str.split('.')
    module_part = '.'.join(func_tokens[0:-1]).strip()
    func_part = func_obj.__name__

    if type(func_obj).__name__ in ('builtin_function_or_method', 'function'):
      if not module_part:
        return '%s.%s'%(func_obj.__module__, func_part)

      elif func_obj.__module__:
        return '%s.%s'%(func_obj.__module__, func_part)

      else:
        module_obj = eval(module_part, {}, globals_)
        if hasattr(module_obj, '__name__'):
          return '%s.%s'%(module_obj.__name__, func_part)

    if hasattr(func_obj, '__self__'):

      if hasattr(func_obj.__self__, '__class__'):
        class_obj = func_obj.__self__.__class__
      else:
        class_obj = func_obj.__self__

      method_name = self.tryFindCompatibleMethodName(class_obj, func_obj)
      return method_name

    raise ValueError('Unbound Method: %s', func_str)

  def tryFindCompatibleMethodName(self, class_obj, func_obj):
    name_orig = '%s.%s'%(self.getClassObjectName(class_obj), func_obj.__name__)
    if name_orig in self.func_ids:
      return name_orig

    arr = [class_obj]
    seen_set = set()
    while arr:
      cur_class_obj = arr.pop(0)
      seen_set.add(cur_class_obj)
      name = '%s.%s'%(self.getClassObjectName(cur_class_obj), func_obj.__name__)
      if name in self.func_ids:
        return name
      if self.stripModuleComponent(name) in self.func_ids:
        return self.stripModuleComponent(name)
      bases = filter(lambda x: x != object and x not in seen_set,  cur_class_obj.__bases__)
      if bases:
        arr.extend(list(bases))

    if name_orig.startswith('matplotlib'):
      print '\n  ** Failed to find API-compatible name for %s\n'%name_orig
    return name_orig

  def getClassObjectName(self, class_obj):
    return '%s.%s'%(class_obj.__module__, class_obj.__name__)

  def stripModuleComponent(self, name):
    tokens = name.split('.')
    module_tokens = filter(lambda x: not x.startswith('_'), tokens[0:-1])
    return '%s.%s'%('.'.join(module_tokens), tokens[-1])

if __name__ == '__main__':

  cnz = CodeAnalyzer()

  def _test_code_from_file(fname):
    with open(fname) as reader:
      code = reader.read()
    codeLines = code.split('\n')
    for func_id, start, end, keywords in cnz.findFunctionsFromCodeBlock(code):
      print func_id, start, end, [_keyword_printer(x, codeLines) for x in keywords]
    print '--------------'

  def _keyword_printer(keyword, codeLines):
    assert keyword.lineno == keyword.end_lineno
    assert codeLines[keyword.lineno-1][keyword.col_offset:keyword.end_col_offset] == keyword.arg
    return keyword.arg

  _test_code_from_file('code-samples/formattest.py')
  _test_code_from_file('code-samples/casestudy2.py')
  try:
    _test_code_from_file('code-samples/casestudy1.py')
  except NameError, e:
    pass
