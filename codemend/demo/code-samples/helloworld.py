import numpy as np
import matplotlib.pyplot as plt

a = np.array([1,2,3])
b = [2,3,1]
c = [2,6,2]

plt.bar(a - 0.3, b, 0.3, color='red', label='A')
plt.bar(a, c, 0.3, color='blue', label='B')
plt.xticks([1,2,3], ['foo','bar','baz'])
