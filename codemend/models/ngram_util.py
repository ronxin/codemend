def ngram_partition(s, word_lookup, func_lookup=None,
                    nl_stopword_set=None, maxngram=5):
  """
  Given an input string, partition the tokens using lookup. The input string
  is assumed to be properly normalized (lower-casing and tokenized).

  @func_lookup - a dictionary, if provided, will replace any token that
  matches a key to the matched value

  @returns a list of ngram wids Designed to be used against question titles,
      and user-input queries not to be used for general S/O posts (additional
      cleaning is needed, see extract_text.py)
  """
  tokens = s.split()
  if func_lookup:
    for ti,t in enumerate(tokens):
      if t in func_lookup:
        tokens[ti] = func_lookup[t]

  out_ngrams = []
  i = 0
  while i < len(tokens):
    j = min(i + maxngram, len(tokens))
    while j >= i + 1:
      s_ngram = '_'.join(tokens[i:j])
      # print i,j,s_ngram
      if s_ngram in word_lookup:
        out_ngrams.append(s_ngram)
        break
      j -= 1
    i = max(j, i+1)

  if nl_stopword_set:
    out_ngrams = filter(lambda x: x not in nl_stopword_set, out_ngrams)

  return out_ngrams

if __name__ == '__main__':
  """
  Test ngram_partition
  """
  s = 'a b c d e f h'
  word_lookup = {
    'a_b': 1,
    'a_b_c': 1,
    'd': 1,
    'f_h': 1,
    'h': 1
  }
  print ngram_partition(s, word_lookup, maxngram=3)
  print "Expects: [a_b_c, d, f_h]"
