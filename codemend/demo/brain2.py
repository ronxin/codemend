from __future__ import division

import traceback
import ast

from collections import defaultdict

from codemend import relative_path
from codemend.demo import server_util
from codemend.demo.code_suggest import CodeSuggest
from codemend.demo.softmax import softmaxed_weights_for_highlight
from codemend.docstring_parse import doc_serve
from codemend.models.annotate_code_with_api import get_fu_fau
from codemend.models.bimodal_baseline2 import BiModalBaseline
from codemend.models.eval2 import ContextBuilder
from codemend.models.word2vec_util import load_gensim_from_binary_file
from codemend.thonny import ast_utils

class Brain:
  def __init__(self):
    print 'Brain initializing ...'
    self.fu, self.fau = get_fu_fau()

    self.cb = ContextBuilder()
    self.w2v = load_gensim_from_binary_file(
      relative_path('models/output/vectors-so-text-python-lemma.bin'))

    self.current_code = None
    self.current_code_hash = None

    self.bimodal = BiModalBaseline('bimodal',
      relative_path('models/output/bi2-0410-d.model'),
      self.w2v)

    self.cs = CodeSuggest()

    self.gws_cache = {}

    print 'Brain intialized.'

  def prepare_current_code(self):
    if self.current_code \
        and hash(self.current_code) != self.current_code_hash:

      self.current_code_hash = hash(self.current_code)
      self.current_node = ast.parse(self.current_code)
      try:
        ast_utils.mark_text_ranges(self.current_node, unicode(self.current_code))
      except AssertionError:
        raise SyntaxError
      except IndexError:
        raise SyntaxError

      self.code_context = self.cb.getContext(self.current_node)

  def handleRequest(self, params, server_handle=None):
    request_type = ''

    try:
      experiment_mode = 'no_mode'
      try:
        with open(relative_path('demo/log-mode.txt')) as reader:
          experiment_mode = reader.read().strip()
      except:
        pass

      request_type = params['type'] if 'type' in params else ''
      ret = {'type': request_type}

      code = params['code'] if 'code' in params else ''
      query = params['query'].strip().lower() if 'query' in params else ''

      self.current_code = code
      self.prepare_current_code()

      control_group = False
      if experiment_mode[0] == 'g':
        control_group = True

      if request_type == 'nlp':
        if not control_group:
          matches = self.get_matches(query)
          ret['matches'] = matches

      elif request_type in ('summary', 'suggest'):
        if not control_group:
          cursor_line = int(params['cursor_line']) + 1
          cursor_ch = int(params['cursor_ch'])
          if request_type == 'summary':
            ret['summary_groups'] = self.cs.get_summary(query, code, cursor_line, cursor_ch, self)
          else:
            elem_id = params['elem_id']
            ret['suggest'] = self.cs.get_suggest(query, code, cursor_line, cursor_ch, self, elem_id)

      elif request_type == 'google':
        pass

      elif request_type == 'experiment_mode':
        ret['mode'] = experiment_mode

      else:
        raise ValueError('Unrecognized request type: "%s"'%request_type)

      return ret

    except SyntaxError:
      return {'error': 'syntax error', 'type': request_type}
    except Exception as e:
      print '\n\nBrain Error:'
      print traceback.format_exc() + '\n\n'
      return {'error': server_util.pack_exception_for_html(e, 'Brain Error'),
              'type': request_type
             }

  def get_matches(self, query):
    if not query.strip(): return []
    if not self.current_code: return []
    if not hasattr(self, 'code_context'): return []

    used_elems = self.code_context.used_elems()
    used_elems_set = set(used_elems)

    suggest_items = self.bimodal.suggest(query, self.code_context)

    elem_scores = softmaxed_weights_for_highlight(suggest_items)
    elem_score_lookup = defaultdict(float)
    for elem, score in elem_scores:
      elem_score_lookup[elem] = score

    for elem, score in elem_score_lookup.items():
      if elem not in used_elems_set or \
          doc_serve.is_positional_argument(elem):
        for parent in doc_serve.find_parents(elem):
          if parent in used_elems_set:
            elem_score_lookup[parent] += score

    used_elem_objs = self.code_context.used_elem_objs()

    matches = []
    for elem_obj in used_elem_objs:
      elem_id = elem_obj.elem
      if not elem_id in elem_score_lookup: continue
      score = elem_score_lookup[elem_id]

      elem_doc = doc_serve.find_elem(elem_id)
      if not elem_doc: continue

      lineno, col_offset, end_lineno, end_col_offset = elem_obj.getTextRange()

      matches.append({
          'weight': score,
          'type': elem_doc.type,
          'elem_id': elem_id,
          'lineno': lineno - 1,
          'end_lineno': end_lineno - 1,
          'col_offset': col_offset,
          'end_col_offset': end_col_offset
        })
    matches = sorted(matches, key=lambda x:len(x['elem_id']))
    return matches
