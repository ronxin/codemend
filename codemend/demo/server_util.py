import sys
import socket
import urllib
import traceback

def port_available_or_die(port):
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  result = s.connect_ex(('127.0.0.1', port))
  s.close()
  if result == 0:
    sys.exit('Port %d is in use!'%port)

def is_url_reachable(url):
  try:
    urllib.urlopen(url).read()
    return True
  except Exception:
    return False

def pack_exception_for_html(exception, error_title, simplified=False):
  """Given an exception, pack it as a HTML-formatted string,
  to be placed in a DIV."""
  traceback_error = traceback.format_exc()
  try:
    idx = traceback_error.index('File "<string>"')
    traceback_error = traceback_error[idx:].replace(', in <module>\n', '\n').replace('File "<string>", ', '')
  except ValueError:
    pass
  return '%s:<br><pre>%s</pre>'%(error_title, traceback_error)
