"""
Extracts text from stack overflow dump.

Unlike extract_so_text.py, this one does not rely on the pickle generted by
extract_so_code.py.

Usage:
  python extract_so_text_all.py  all|python

  all - extract text from all threads
  python - extract text from threads tagged with python only
"""
import sys

from extract_so_text import thread_to_clean_text
from extract_so_code import load_threads, Thread, Answer

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print(globals()['__doc__'] % locals())
    sys.exit(1)

  assert sys.argv[1] in ('all', 'python')

  if sys.argv[1] == 'all':
    qfilter = None
    afilter = None
  else:
    assert sys.argv[1] == 'python'
    qfilter="Tags LIKE '%<python>%'"
    afilter=None  # no need to filter answers in any case

  output_file_name = 'output/so-text-%s.txt'%sys.argv[1]

  count_threads = 0
  with open(output_file_name, 'w') as writer:
    for thread in load_threads(qfilter=qfilter, afilter=afilter):
      count_threads += 1
      if count_threads % 1000 == 0: print 'processed %dK threads'%(count_threads/1000)
      s = thread_to_clean_text(thread).strip()
      writer.write(s.encode('utf-8') + '\n')

  print 'total %d threads'%count_threads
  print 'done.'
