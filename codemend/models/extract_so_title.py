from codemend.models.extract_so_code import load_threads, Thread, Answer
from codemend import BackupHandler, relative_path

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

  with open(relative_path('models/output/mpl_so_titles.txt'), 'w') as writer:
    for t in threads:
      writer.write('%d\t%s\n'%(t.qid, t.qtitle.encode('utf-8')))
