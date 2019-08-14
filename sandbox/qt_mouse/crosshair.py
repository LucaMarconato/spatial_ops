# import matplotlib.pyplot as plt
#
# def cross_hair(x, y, ax=None, **kwargs):
#     if ax is None:
#         ax = plt.gca()
#     horiz = ax.axhline(y, **kwargs)
#     vert = ax.axvline(x, **kwargs)
#     return horiz, vert
#
# cross_hair(0.2, 0.3, color='red')
# plt.show()


from matplotlib.widgets import Cursor
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

x, y = 4*(np.random.rand(2, 100)-.5)
ax.plot(x, y, 'o')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# set useblit = True on gtkagg for enhanced performance
cursor = Cursor(ax, useblit=True, color='red', linewidth=2 )
plt.show()

