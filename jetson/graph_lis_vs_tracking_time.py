import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


objects = ('Tracking', 'LIS')
y_pos = np.arange(len(objects))
performance = [0.4122353304004719, 0.7437010366498725]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Time [s]')
plt.title('Average time for attitude estimation: Tracking vs LIS modes')

plt.show()
