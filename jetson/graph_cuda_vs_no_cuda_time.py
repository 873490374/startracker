import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


objects = ('GPU (CUDA)', 'CPU')
y_pos = np.arange(len(objects))
performance = [0.06747964299938758, 2.1326403450002545]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Time [s]')
plt.title('Star recognition (centroid calculation): GPU vs CPU')

plt.show()
