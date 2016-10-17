"""
Polish documentation so as to:
1. Support return type tracking.
2. To have both short utterance (used for machine interpretation) and longer-
   verion descriptions.
3. Support documentation to dictionary key.

The polishing process is additive, in that:
- Minimal modification to existing (complicated) pipeline.
- Supports manual labeling input by human, and the software will do a multi-
  way merge: merging multiple existing sources together with human input to
  obtain a polished doc for instant inspection.

Existing sources:
- Database that contains extracted documentations.
- Simplified fu and fau CSV files (reference only).
- Element counts from 15K SO code examples and 8K GitHub code files.

Human annotation (including what is coded into this script) provides:
- Missing parameters.
- Deletions for faulty parameters
- Dictionary keys (including reference pointers)
- Positional argument combinations (for inferring which is which).

Algorithm:
1. Create faku and augment fau using additional_ref.csv
2. Go through every element in element counts file.
3. Try to match doc and everything else for them.
4. Use additional_doc.csv to augment corner cases.

"""

import csv
import pydoc
from itertools import imap
from collections import defaultdict, namedtuple
from recordclass import recordclass
import funcsigs
import string

from codemend import BackupHandler, relative_path
from codemend.models.element import ElementNormalizer
from codemend.docstring_parse.elemdoc import ElemDoc
from codemend.docstring_parse.consolidate import is_setXXX, get_class
from codemend.models.annotate_code_with_api import get_fu_fau

# Load all input sources
bh = BackupHandler(relative_path('experimental/code_suggest/output/backup'))

fu_t, fau_t = get_fu_fau(omit_module=False, truncate=True)
fu, fau = get_fu_fau(omit_module=False, truncate=False)

fa_lookup = defaultdict(list)  # [function] = [argument]
for f, a in fau.keys():
  fa_lookup[f].append(a)
fa_lookup = dict(fa_lookup)

cf_lookup = defaultdict(list)  # [class] = [function]
for f in fu.keys():
  cf_lookup[get_class(f)].append(f)

elem_counts = bh.load('elem_pyplot_counts_0404')
enormer = ElementNormalizer()

rtype_lookup = {}  # [elem_id] = rtype (simplified)
with open(relative_path(
    'docstring_parse/annotation/rtype_map.csv'), 'rb') as csvfile:
  reader = csv.reader(csvfile)
  for elem_id, rtype in reader:
    rtype_lookup[elem_id] = enormer.unsimplify(rtype)

adoc_lookup = {}  # [elem_id] = utter (additional doc)
with open(relative_path(
    'docstring_parse/annotation/additional_doc.csv'), 'rb') as csvfile:
  reader = csv.reader(csvfile)
  ADoc = recordclass('ADoc', next(reader))
  assert ADoc._fields == ('elem_id', 'doc')
  for adoc in imap(ADoc._make, reader):
    adoc_lookup[adoc.elem_id] = adoc.doc

aref_lookup = defaultdict(set)  # [elem_id] = [ARef] (additional reference)
with open(relative_path(
    'docstring_parse/annotation/additional_ref.csv'), 'rb') as csvfile:
  reader = csv.reader(csvfile)
  ARef = namedtuple('ARef', next(reader))
  assert ARef._fields == ('elem_id', 'ref', 'type')
  for aref in imap(ARef._make, reader):
    aref_lookup[aref.elem_id].add(aref)

# Use aref to augment fau and create faku
faku = {}  # [f@a,k] = utter
faku_t = {}  # truncated
def complain_cf_lookup(target, ref):
  if not target in cf_lookup:
    print 'ERROR in addition_ref.csv: %s (%s) not a seen class in fu.csv'%(target, ref)
    return True
  return False
def complain_fa_lookup(target, ref):
  if not target in fa_lookup:
    print 'ERROR in addition_ref.csv: %s (%s) not a seen func in fau.csv'%(target, ref)
    return True
  return False
for iter_count in xrange(3):
  for arefs in aref_lookup.values():  # each elem_id may have multiple entries
    for aref in arefs:
      source = enormer.unsimplify(aref.elem_id)
      target = enormer.unsimplify(aref.ref)
      if '@' in source:
        # source is an argument
        if aref.type == 'property_as_keyword':
          # borrow all of target's properties as my arg's dict key
          if complain_cf_lookup(target, aref.ref): continue
          for f in cf_lookup[target]:
            property_ = is_setXXX(f)
            if property_ and (source, property_) not in faku:
              faku[source, property_] = fu[f]
              faku_t[source, property_] = fu_t[f]
        else:
          # borrow all of target's keywords as may arg's dict key
          assert aref.type == 'keyword_as_keyword', aref.type
          if complain_fa_lookup(target, aref.ref): continue
          for a in fa_lookup[target]:
            if (source, a) not in faku:
              faku[source, a] = fau[target, a]
              faku_t[source, a] = fau_t[target, a]
      else:
        # source is a function
        if aref.type == 'property_as_keyword':
          # borrow all of target's properties as my keyword arguments
          if complain_cf_lookup(target, aref.ref): continue
          for f in cf_lookup[target]:
            property_ = is_setXXX(f)
            if property_ and (source, property_) not in fau:
              fau[source, property_] = fu[f]
              fau_t[source, property_] = fu_t[f]
              if enormer.simplify(f) in aref_lookup and iter_count == 0:
                # e.g., mpl.text.Text.set_bbox@... --> plt.title@bbox@...
                # Put this secondary ref into aref_lookup to be processed in
                # the next round.
                for aref_2 in aref_lookup[enormer.simplify(f)]:
                  new_aref = ARef(aref.elem_id+'@'+property_, aref_2.ref, aref_2.type)
                  aref_lookup[new_aref.elem_id].add(new_aref)
        else:
          assert aref.type == 'keyword_as_keyword', aref.type
          # borrow all of target's arguments as my arguments
          if complain_fa_lookup(target, aref.ref): continue
          for a in fa_lookup[target]:
            if (source, a) not in fau:
              fau[source, a] = fau[target, a]
              fau_t[source, a] = fau_t[target, a]

def create_new_element_doc(elem_id):
  assert elem_id
  fields = elem_id.split('@')
  fields2 = fields[0].split('.')

  full_name = enormer.unsimplify(enormer.infer(elem_id, False))

  utter = ''
  utter_expand = ''
  assert len(fields) >= 1 and len(fields) <= 3

  if len(fields) == 3:
    type_ = 'argkey'
    name = fields[-1]
    parent_id = '@'.join(fields[:-1])
    parent_full_name = enormer.unsimplify(enormer.infer(parent_id, False))
    if (parent_full_name, name) in faku:
      utter = faku_t[parent_full_name, name]  # truncated
      utter_expand = faku[parent_full_name, name]
      # print 'Found', (parent_full_name, name)
    else:
      # print 'Not found', (parent_full_name, name)
      pass

  elif len(fields) == 2:
    type_ = 'arg'
    name = fields[-1]
    if not name \
        or name.lower() in ('true', 'false') \
        or ' ' in name:
      # invalid argument name
      return None
    parent_id = fields[0]
    parent_full_name = enormer.unsimplify(enormer.infer(parent_id, False))

    """
    if name[0] in string.digits:  # try to interpret positional argument
      thing = pydoc.locate(parent_full_name)
      if thing:
        try:
          signature = funcsigs.signature(thing)
          name_int = int(name)
          params = signature.parameters.keys()
          if name_int < len(params):
            name = params[name_int]
        except ValueError:
          pass
        except TypeError:
          pass
    """

    if (parent_full_name, name) in fau:
      utter = fau_t[parent_full_name, name]  # truncated
      utter_expand = fau[parent_full_name, name]

  else:
    assert len(fields) == 1
    type_ = 'func'
    name = fields2[-1]
    parent_id = '.'.join(fields2[:-1])
    if full_name in fu:
      utter = fu_t[full_name]  # truncated
      utter_expand = fu[full_name]
    else:
      pass
      # thing = pydoc.locate(full_name)
      # if thing:
      #   utter = pydoc.describe(thing)
      #   utter_expand = pydoc.html.document(thing)

  if not utter:
    if elem_id in adoc_lookup:  # additional_doc.csv
      utter = adoc_lookup[elem_id]
      utter_expand = adoc_lookup[elem_id]

  rtype = ''
  if elem_id in rtype_lookup:
    rtype = rtype_lookup[elem_id]  # already unsimplified

  count = 0
  if elem_id in elem_counts:
    count = elem_counts[elem_id]

  elem = ElemDoc(elem_id, name, full_name, type_, parent_id,
                     rtype, count, utter, utter_expand)
  return elem

if __name__ == '__main__':
  # Iterate through all element IDs and create ElemDoc objects.
  candidate_elem_ids = set(elem_counts.keys())
  candidate_elem_ids |= set(enormer.simplify(f) for f in fu.keys())
  candidate_elem_ids |= set(enormer.simplify(f) + '@' + a for f, a in fau.keys())
  candidate_elem_ids |= set(adoc_lookup.keys())

  all_elems = []
  for elem_id in candidate_elem_ids:
    elem = create_new_element_doc(elem_id)
    if elem: all_elems.append(elem)

  # Write to Output
  with open(relative_path(
      'docstring_parse/doc_polished/elem_docs.csv'), 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(ElemDoc._fields)
    writer.writerows(all_elems)

  print 'Done'
