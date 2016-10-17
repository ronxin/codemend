import numpy as np
import matplotlib.pyplot as plt
a = [0.62, 0.95, 0.85]
b = [0.38, 0.62, 0.82]
data = np.array([a,b])
people = ['Adam', 'Bobby', 'Charlie']
color_map = plt.get_cmap('Pastel1', 9)
labels = ['Before', 'After']
patterns = ['/','\\']
for group in range(2):
    left = np.arange(data.shape[1])*3 + group
    height = data[group]
    color = color_map(group)
    bars = plt.bar(left, height, color=color)
    for bar in bars:
        bar.set_width(0.8)
        if bars.index(bar) == 0:
            bar.set_label(labels[group])
        bar.set_hatch(patterns[group])
plt.xlim(-1,9)
plt.xticks([1,4,7], people)
plt.legend(loc='lower center', ncol=2, shadow=True,
           bbox_to_anchor=(0.5,-.2))
plt.title('Experiment Result')
