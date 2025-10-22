import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0,10,11)
p = np.ones(len(theta)) * 1/(len(theta) + 1)


plt.stem(theta,p)
plt.ylim([0,0.2])
plt.show()