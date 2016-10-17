import numpy as np
import matplotlib.pyplot as plt

N = 50
np.random.seed(1)
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2

plt.figure(figsize=(8, 6))
plt.scatter(x, y, s=area, c=colors, alpha=0.5)

i = np.argmax(area)
arrow_end = xy=(x[i],y[i])
arrow_start = (x[i]+0.1,y[i]+0.1)
plt.annotate('biggest', arrow_end,
             xytext=arrow_start,
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.ylim(-0.5, 1.5)
plt.xlabel('X')
plt.ylabel('Y')
