import string
import pattern.en
import pattern.vector

def tokenize(s):
  return ' '.join(pattern.en.tokenize(s, replace={'-': ' '})).split()

def lemma(token):
  return pattern.vector.stem(token, stemmer=pattern.vector.LEMMA)

def lemmaWithVocab(token, vocab):
  """
  First try PORTER. If not in vocab, then lemma.

  """
  out = pattern.vector.stem(token, stemmer=pattern.vector.PORTER)
  if out in vocab: return out
  if token.endswith('ing'): out = token[:-3]
  if out in vocab: return out
  return lemma(token)

from codemend import relative_path

stopwords = set()
with open(relative_path('models/stopwords-en.txt')) as reader:
  for line in reader:
    if line.startswith('#'): continue
    line = line.strip()
    words = line.split(', ')
    stopwords |= set(words)
  stopwords |= set(string.punctuation)


if __name__ == '__main__':
  s = 'hatching-hatches color" colors interesting flying flies'
  vocab = dict(hatch=1, color=2, interest=3, fly=4)
  tokens = tokenize(s)
  lemmas = [lemmaWithVocab(x, vocab) for x in tokens]
  print lemmas
