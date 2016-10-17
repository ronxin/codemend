"""
Extracts text from Stack Overflow posts. Code snippets are excluded.

The text corpus extracted is to be used for training a flat-word2vec model.
Tokenization and cleaning are performed.

Requires mpl_threads.pickle, which can be generated using extract_so_code.py

Procedure:
 - Go through each thread in SO tagged matplotlib
 - Extract title and content from question and answers
 - Clean up so that only good text remains.
   - replace HTML escaped characters
   - remove code snippets
   - remove other tags
   - tokenize
 - Dump them to a text file.
"""
import os
import re
import sys

"""
To install, go to https://pypi.python.org/pypi/Markdown download, unzip, and
run "pip install --user markdown" on a server, or without "--user" if has root
"""
import markdown

# These are consim libs. See consim/py/lib/README
from backup_util import BackupHandler
import html2plain
from nltk.tokenize import word_tokenize, sent_tokenize

# Refer to consim/py/acl/wiki_util.py for more examples
# This one is directly copied from there
def strip_tags(tag_name, input_string):
  assert isinstance(input_string, basestring)
  return re.sub(r'<%s.*?>.*?</%s>'%(tag_name, tag_name), '', input_string, flags=re.DOTALL)

def strip_code(input_string):
  return strip_tags('code', input_string)

def markdown2html(input_string):
  return markdown.markdown(input_string)

def html2plaintext(input_string):
  return html2plain.strip_tags(input_string)

def tokenize(s):
  # return word_tokenize(s)
  # We have to first break into sentences, then break each sentence into tokens, in order for word_tokenizer to function correctly
  # To download proper files for sent_tokenize(), refer to http://stackoverflow.com/questions/4867197/failed-loading-english-pickle-with-nltk-data-load
  s = ' '.join(map(lambda x:' '.join(word_tokenize(x)), sent_tokenize(s)))
  # following are supplementary tokenization
  s = re.sub(r"'(\w\w\w+)", r"' \1", s)  # "'somethign" -> "' something" - excluding short ones, such as 's, 've, etc.
  s = re.sub(r'(\w)\.\.', r'\1 ..', s)  # "somethign.." -> "something .."
  s = re.sub(r'(\w)/(\w)', r'\1 / \2', s)  # "and/or" -> "and / or"
  return s

def dirty2clean(s):
  # this call must be before html2text, otherwise the '<code>' tag would be
  # removed
  s = strip_code(s)
  s = html2plaintext(s)
  s = markdown2html(s)
  # I am not crazy. This is b/c s/o text is mixed with markdown and html.
  s = html2plaintext(s)
  s = re.sub(r'\s+', ' ', s)
  s = tokenize(s)
  s = s.lower()
  return s

def thread_to_clean_text(t):
  s = '\n'.join([t.qtitle, t.qbody] + [x.abody for x in t.answers])
  s = dirty2clean(s).strip()
  return s

if __name__ == '__main__':
  bh_dir = 'output/backup'
  if not os.path.exists(bh_dir):
    os.makedirs(bh_dir)
  bh = BackupHandler(bh_dir)

  from extract_so_code import Thread, Answer
  try:
    threads = bh.load('mpl_threads')
  except AssertionError:
    print 'Need to use extract_so_code.py to extract mpl_threads first.'
    sys.exit(1)

  with open('output/mpl_so_flat_text.txt', 'w') as writer:
    for ti, t in enumerate(threads):
      if (ti + 1) % 1000 == 0: print '%d / %d'%(ti+1, len(threads))
      s = thread_to_clean_text(t)
      writer.write(s.encode('utf-8') + '\n')
