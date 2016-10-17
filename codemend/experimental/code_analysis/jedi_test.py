import jedi

source = '''import matplotlib.pyplot as plt
: : some syntax error this line
plt.bar(
'''

script = jedi.Script(source, 3, len('plt.bar('))

print script.call_signatures()

completions = script.completions()
for c in completions:
  print c
