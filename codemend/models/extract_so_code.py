"""
Script to extract code snippets from a stack overflow dump.

Code taken from the following files in consim/py/so/
 - extract_thread.py
 - prepare_ann.py
 - extract_text.py
 - extract_goals.py
"""

import sqlite3
from collections import namedtuple
import os

from codemend import BackupHandler, relative_path

Thread = namedtuple('Thread', ['qid', 'qtitle', 'qbody', 'qscore', 'answers'])
Answer = namedtuple('Answer', ['aid', 'abody', 'ascore'])

def extract_code_blocks(s):
  """
  s is a string, representing the content body of a post in its raw format.

  Returns a list of strings. Each string is an extracted code block.
  """
  assert isinstance(s, basestring)
  prefix = '<pre><code>'
  suffix = '</code></pre>'
  fs = s.split(prefix)
  blocks = []
  for f in fs[1:]:
    fs2 = f.split(suffix, 1)
    if len(fs2) > 1:
      blocks.append(fs2[0])
  return blocks


def load_threads(qfilter=None, afilter=None):
  """
  Find Stack Overflow posts
     (a) containing <matplotlib> tag (with optionally other tags)
     (b) #answer_with_positive_score >= 1
     (c) (answer) among top 3 ranked answer

  Using the following data structure:
     thread = [(qid, qtitle, qbody, qscore, answers)]
     answers = [(aid, abody, ascore)]
     qid = question id
     aid = answer id

  Yields a sequence of Threads
  """
  db = sqlite3.connect('/storage6/users/ronxin/work/db/so-dump.db')
  cursor = db.cursor()
  q = "SELECT Id, Title, Body, Score from posts"
  if qfilter:
    q = '%s WHERE %s'%(q, qfilter)
  cursor.execute(q)
  count = 0
  for row in cursor:
    count += 1
    if count % 1000 == 0: print count
    qid, qtitle, qbody, qscore = row
    q2 = "SELECT Id, Body, Score from posts WHERE ParentID = %d"%qid
    if afilter:
      q2 = '%s AND %s'%(q2, afilter)
    answers = []
    cursor2 = db.cursor()
    cursor2 = cursor2.execute(q2)
    answers = map(Answer._make, cursor2.fetchall())
    if not answers: continue
    answers = tuple(answers)
    yield Thread._make(row + (answers,))

if __name__ == '__main__':
  bh_dir = relative_path('models/output/backup')
  bh = BackupHandler(bh_dir)

  try:
    threads = bh.load('mpl_threads')
  except AssertionError:
    threads = list(load_threads(
      qfilter="Tags LIKE '%<matplotlib>%' AND AnswerCount > 0 AND Score >= 0",
      afilter="Score >= 0 ORDER BY Score DESC LIMIT 3"))
    bh.save('mpl_threads', threads)

  # Dump the code blocks extracted from the threads
  print 'Extracting and dumping code blocks ...'
  outdir = 'output/'
  if not os.path.exists(outdir):
    os.makedirs(outdir)

  with open(os.path.join(outdir, 'mpl_code_blocks.txt'), 'w') as writer:
    total_threads = 0
    total_answers = 0
    total_code_blocks = 0
    for thread in threads:
      total_threads += 1
      for answer in thread.answers:
        total_answers += 1
        for code_block in extract_code_blocks(answer.abody):
          total_code_blocks += 1
          writer.write('%s\n\n'%code_block.encode('utf-8'))

  print 'Done.'
  print 'total_threads', total_threads
  print 'total_answers', total_answers
  print 'total_code_blocks', total_code_blocks
