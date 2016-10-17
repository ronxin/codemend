import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

y1 = [10.1,7.95,7.0,6.5,6.3,6.1,6.0,5.8,5.75,5.5,5.45,5.4,5.3,5.2,5.18,5.1,5.12,5.1,5.08,5.01,5.8]
y2 = [9.5,4.5,8.0,6,4,7,5,5.5,5.5,6,7,5.5,4.5,4,4,4.5,5.5,4.5,5,5,5]
xticklabels = '20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40+'.split()
yticks = range(4,11)
xlabel = 'Solution Popularity (support)'
ylabel = 'Solutions (Itemset) Size'
curveLabels = ['Average', 'Median']  # y1, y2

font = {'family': 'sans-serif',
        'size': 15,
       }
mpl.rc('font', **font)
plt.figure(figsize=(9,4))
plt.plot(y1, '-D', label=curveLabels[0], color='#555555', lw=2, markeredgecolor='None', markersize=6, zorder=10, clip_on=False)
plt.plot(y2, '-s', label=curveLabels[1], color='#AAAAAA', lw=2, markeredgecolor='None', markersize=8, zorder=11, clip_on=False)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.yticks(yticks)
plt.xticks(np.arange(len(y1)), xticklabels, rotation='45')
legend = plt.legend(fontsize=16, bbox_to_anchor=(0.95,0.955), numpoints=1, borderpad=0.9, handlelength=3)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')
plt.gca().xaxis.grid(False)
plt.gca().yaxis.grid(True, color='black', linestyle='-')
plt.xlim(-0.5, len(y1)-0.5)
plt.ylim(4, 10.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().get_xaxis().tick_bottom()
plt.gca().get_yaxis().tick_left()
for tic in plt.gca().xaxis.get_major_ticks():
    tic.tick1On = tic.tick2On = False
plt.gca().tick_params(axis='y', direction='out')
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['left'].set_color('#888888')
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['bottom'].set_color('#888888')
plt.gca().yaxis.set_tick_params(width=2, length=5, color='#888888')
