import sys

def main():
  """
  Usage:

    check_bimodal file "query"

  """

if __name__ == '__main__':
  if len(sys.argv) < 3:
    print main.__doc__
    sys.exit(1)

  filename = sys.argv[1]
  query = sys.argv[2]
  if not filename in ('bar', 'pie', 'line'):
    print 'unrecognized file id'
    sys.exit(1)

  from codemend.models.test_bimodal2 import load_model, eval_one

  w2v_model, model = load_model()
  _, _, suggest_items = eval_one(model, True, '', query, filename)

  max_items = 20
  for item, score in suggest_items[:max_items]:
    print '%f\t%s'%(score, item)
