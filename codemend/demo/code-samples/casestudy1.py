import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# F1
common1 = [0.62, 0.95, 0.85]
longtail1 = [0.38, 0.62, 0.82]
baselines1 = ['Seal', 'SealDict', 'Lyretail']
ylabel1 = 'F1'
title1 = 'Page-specific Extraction'
xtickLabels1 = ['Common Vocabulary', 'Longtail Vocabulary']

# MAP
common2 = [0.98, 0.99, 1.0]
longtail2 = [0.12, 0.6, 0.67]
baselines2 = ['Seisa', 'Seal', 'Lyretail']
ylabel2 = 'MAP'
title2 = 'Dictionary generation'


def plotOneAx(ax, barsLeft, barsRight, xtickLabels, ylabel, subtitle, alphaTitle, baselines):
    NLeft = len(barsLeft)
    NRight = len(barsRight)
    barWidth = 1
    xLeft = np.arange(NLeft)
    xRight = np.arange(NLeft+1, NLeft+1+NRight)

    cm = plt.get_cmap('Pastel1', 9)  # Tricky! setting it to 3 doesn't look good
    hatch_patterns = ['/', '.', '\\']
    bars = []
    font = {'family': 'serif',
            'size': 18,
            'serif': 'Times New Roman',
           }
    matplotlib.rc('font', **font)
    for i in range(len(xLeft)):
        b = ax.bar(xLeft[i], barsLeft[i], barWidth, color=cm(i), hatch=hatch_patterns[i])
        bars.append(b)
    for i in range(len(xRight)):
        b = ax.bar(xRight[i], barsRight[i], barWidth, color=cm(i), hatch=hatch_patterns[i])
        bars.append(b)

    xtickLabels = map(lambda x:x.replace(' ', '\n'), xtickLabels)

    ax.set_ylabel(ylabel)
    ax.set_xticks([np.median(xLeft) + 0.5 * barWidth, np.median(xRight) + 0.5 * barWidth])
    ax.set_xticklabels(xtickLabels)
    ax.set_title(subtitle)
    ax.legend(bars[:NLeft], baselines, loc='lower center',
              ncol=len(baselines), bbox_to_anchor=(0.5,-.36), prop={'size':16},
              shadow=True)
    ax.set_xlim(ax.get_xlim()[0] - 0.5 * barWidth, ax.get_xlim()[1] + 0.5 * barWidth)
    ax.text(ax.get_xlim()[1]/2,-0.4,alphaTitle, {'size':18})

fig, axes = plt.subplots(figsize=(12,4), ncols=2)
plotOneAx(axes[0], common1, longtail1, xtickLabels1, ylabel1, title1, '(a)', baselines1)
plotOneAx(axes[1], common2, longtail2, xtickLabels1, ylabel2, title2, '(b)', baselines2)
