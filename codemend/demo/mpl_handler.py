import numpy as np
import matplotlib
matplotlib.use('Agg')  # prevents window from popping up
import matplotlib.pyplot as plt
import StringIO
import server_util
from pylab import *

from codemend.experimental.code_suggest.mine_examples import issafe

class MplHandler:
  def __init__(self):
    pass

  def handleRequest(self, params):
    try:
      code = params['code'] if 'code' in params else ''
      svg = ''
      if code:
        if not issafe(code):
          return {'error': 'ERROR:<br><pre>Code Not Safe. ' \
          'See codemend.experimental.code_suggest.mine_examples.issafe.</pre>'}
        svg = self.execute(code)
      return {'svg': svg}
    except Exception as e:
      return {'error': server_util.pack_exception_for_html(e, 'Matplotlib Error', simplified=True)}

  def execute(self, code):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as PLT
    import pylab
    matplotlib.rcdefaults()
    imgdata = StringIO.StringIO()
    codeObj = compile(code, '<string>', 'exec')
    exec codeObj
    plt.savefig(imgdata, format='svg', bbox_inches='tight')
    plt.close()
    imgdata.seek(0)
    return imgdata.buf

def main():
  """test"""
  mh = MplHandler()
  result = mh.handleRequest({'code': 'plt.plot([1,2,3],[2,3,1])'})
  print 'Generated SVG of %d chars'%len(result['svg'])
  print 'expecting 10K+ chars'

if __name__ == '__main__':
  main()
