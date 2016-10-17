import os
import cPickle as pickle

class BackupHandler:
  def __init__(self, folder):
    if os.path.exists(folder):
      assert os.path.isdir(folder)
    else:
      os.makedirs(folder)
    self.folder = folder

  def getFileName(self, varname):
    if not varname.endswith('.pickle'): varname = varname + '.pickle'
    return os.path.join(self.folder, varname)

  def save(self, name, value):
    """
    Save anything as a pickle.
    """
    assert isinstance(name, basestring), 'input name needs to be string, not %s'%type(name)
    assert type(value) != type(None), 'input value must not be None'
    p_dict = {'val': value}
    fileName = self.getFileName(name)
    with open(fileName, 'wb') as writer:
      pickle.dump(p_dict, writer)
    print 'Saved to %s'%(fileName)

  def load(self, name):
    """
    Load anything from a pickle.
    """
    fileName = self.getFileName(name)
    assert os.path.isfile(fileName), 'file %s does not exist'%fileName
    with open(fileName, 'rb') as reader:
      p_dict = pickle.load(reader)
    assert len(p_dict) == 1, 'Expecting the p_dict to have only ONE key. This one has %d'%len(p_dict)
    value = p_dict.values()[0]
    print 'Restored from %s'%(fileName)
    return value

  def exists(self, name):
    """
    Returns True if exists. False if not.

    """
    fileName = self.getFileName(name)
    return os.path.isfile(fileName)


def main():
  from codemend import relative_path
  bh = BackupHandler(relative_path('utils/output/backup'))
  a = [1,2,3,{4:[5,6]}]
  bh.save('backup_test_a', a)
  assert bh.exists('backup_test_a')
  b = bh.load('backup_test_a')
  assert a == b
  print 'All test passed.'

if __name__ == '__main__':
  main()
