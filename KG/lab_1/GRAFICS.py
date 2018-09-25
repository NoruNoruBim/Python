'''
Программа построения графика
Кривой (x^2)/(a^2) + (y^2)/(b^2) = 1
Выполнил студент гр. М8о-307 Баранов А.А.
'''

import matplotlib as mpl
import matplotlib.pyplot as plt


print("Write `a` and `b`:")
a = float(input())
b = float(input())

ellipse = mpl.patches.Ellipse(xy = (0, 0), width = a, height = b)
fig, ax = plt.subplots()
fig.gca().add_artist(ellipse)

ax.set_aspect("equal")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

plt.show()
