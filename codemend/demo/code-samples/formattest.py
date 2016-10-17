import numpy as np
import matplotlib.pyplot as plt

a = np.arange(3)
np.random.seed(1)
b = a + np.random.randn(3)

plt. \
  plot(a, b,
  linestyle='--')
plt.xlabel('x')
plt.ylabel("y")

fig, ax = plt.subplots()
ax.plot(a,b)
ax.set_title('hello world')
