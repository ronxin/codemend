import sys

from doc_serve import find_elem, find_children, find_parents

def main():
  """
  Shows all information about an element.

  Usage:
    edoc <elem_id>
      - look up detailed info about this element
    edoc <elem_id>?
      - look up the element's children
    edoc <elem_id>$
      - look up the element's parents

  """

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print main.__doc__
    sys.exit(1)

  elem_id = sys.argv[1]

  if elem_id.endswith('?'):
    elem_id = elem_id.rstrip('?.@')
    children = find_children(elem_id)
    if children:
      children = sorted(children, key=lambda x:x.elem_id)
      for child in children:
        print child.elem_id
    else:
      print 'No children found: %s'%elem_id

  elif elem_id.endswith('$'):
    elem_id = elem_id.rstrip('$')
    parents = find_parents(elem_id)
    if parents:
      for parent in parents:
        print parent
    else:
      print 'No parent found: %s'%elem_id

  else:
    elem = find_elem(elem_id)
    if elem:
      for field in elem._fields:
        if field == 'utter_expand':
          print '[%10s]'%field
          print ''.join(['-'] * (len(field) + 2))
          print getattr(elem, field)
          print
        else:
          print '[%10s] %s'%(field, getattr(elem, field))
    else:
      print 'Elem not found: %s'%elem_id
