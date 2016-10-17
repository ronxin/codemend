import codemend
import os

def relative_path(path):
  return os.path.join(os.path.dirname(codemend.__file__), path)
