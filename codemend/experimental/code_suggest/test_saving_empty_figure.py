"""
Check whether an empty figure will generate a non-empty SVG.
"""
import matplotlib.pyplot as plt
import StringIO

imgdata = StringIO.StringIO()
print 'Number of figure: %d'%len(plt.get_fignums())
plt.savefig(imgdata, format='svg', bbox_inches='tight')
plt.close()
imgdata.seek(0)
svg = imgdata.buf
print 'Length of SVG: %d'%len(svg)
