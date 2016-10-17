import json
import multiprocessing
import os
import SimpleHTTPServer
import SocketServer
import sys
import urlparse
import time

import server_util
import mpl_handler

from codemend import relative_path

# HOST_NAME = '0.0.0.0'
HOST_NAME = '127.0.0.1'
PORT_NUMBER = 9001

mh = mpl_handler.MplHandler()

from brain2 import Brain
my_brain = Brain()

def mh_handle_request(params):
  return mh.handleRequest(params)

class MyRequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
  def do_GET(self):
    def send_header():
      self.send_response(200)
      self.send_header("Content-type", "text/plain")
      self.end_headers()

    def parse_params():
      o = urlparse.urlparse(self.path)
      params = dict(urlparse.parse_qsl(o.query))
      return params

    def send_result(result):
      result_json = json.dumps(result)
      self.wfile.write(result_json)

    if self.path.startswith('/plt-request?'):
      # print 'Received a plt request'
      send_header()
      params = parse_params()
      # uses the shared worker pool
      result = plotter_pool.apply(mh_handle_request, (params,))
      send_result(result)

    elif self.path.startswith('/brain-request?'):
      # print 'Received a brain request'
      send_header()
      params = parse_params()
      result = my_brain.handleRequest(params, self)
      send_result(result)

    else:
      # Regular request
      SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)

  # log_file = open('log-%d.txt'%(int(time.time())), 'w', 1)  # line-buffered
  def log_message(self, format, *args):
    pass
    """
    self.log_file.write("[%s] %s\n" %
                        (self.log_date_time_string(),
                         format%args))
    """

  def do_mh_handle_requests(self, params_list):
    """This callback function is to be used by brain, so that it can take
    advantage of the worker pool to do multithreaded plotting."""
    return plotter_pool.map(mh_handle_request, params_list)

  def jedi_lock(self):
    """To be used by brain to protect non-thread-safe jedi."""
    return jedi_lock

class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
  pass

if __name__ == '__main__':
  os.chdir(relative_path('demo'))

  if len(sys.argv) > 1 and sys.argv[1] != '-':
    port = int(sys.argv[1])
  else:
    port = PORT_NUMBER

  if len(sys.argv) > 2 and sys.argv[2] != '-':
    host_name = sys.argv[2]
  else:
    host_name = HOST_NAME


  server_util.port_available_or_die(port)

  plotter_pool = multiprocessing.Pool()

  # See: http://stackoverflow.com/questions/20742637/
  jedi_lock = multiprocessing.Manager().Lock()
  handler = MyRequestHandler
  server = ThreadedTCPServer((host_name, port), handler)

  print 'Started Serving on %s:%d'%(host_name, port)
  print 'Please open the following URL:\nhttp://localhost:%d\n\n'%port
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    server.server_close()
  print '\nServer Stopped.'
