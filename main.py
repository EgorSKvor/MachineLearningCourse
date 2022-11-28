import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-5, 5, 100)
y = np.sin(x)
plt.figure(figsize=(4, 3))
plt.plot(x, y)
plt.show()