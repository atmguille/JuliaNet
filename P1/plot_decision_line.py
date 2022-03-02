import matplotlib.pyplot as plt
import numpy as np

MODEL_NAME = 'model_name'

w1, w2, b = 1.0, 1.0, 0.0
points = (((1, 1), '+'), ((-1, 1), '+'), ((1, -1), '_'), ((-1, -1), '_'))

x = np.linspace(-1.25, 1.25, 100)
y = (-w1 * x - b) / w2

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.xticks([-1, 1])
plt.yticks([-1, 1])

ax.plot(x, y)

for (x, y), symbol in points:
    ax.plot(x, y, symbol, color='orange', markersize=10, markeredgewidth=2)

plt.savefig(f'{MODEL_NAME}_decision_line.png')
