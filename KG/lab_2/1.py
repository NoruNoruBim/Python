import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def plot_cube(param, color, view):
	points = np.array([[-1, -1, -1],
			  [1, -1, -1 ],
			  [1, 1, -1],
			  [-1, 1, -1],
			  [-1, -1, 1],
			  [1, -1, 1 ],
			  [1, 1, 1],
			  [-1, 1, 1]])
	points *= param

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')


	# plot vertices
	ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])

	# list of sides' polygons of figure
	verts = [[points[0],points[1],points[2],points[3]],
	 [points[4],points[5],points[6],points[7]], 
	 [points[0],points[1],points[5],points[4]], 
	 [points[2],points[3],points[7],points[6]], 
	 [points[1],points[2],points[6],points[5]],
	 [points[4],points[7],points[3],points[0]], 
	 [points[2],points[3],points[7],points[6]]]

	# plot sides
	if color == 0:
		cl = (0, 0, 0, view)
	if color == 1:
		cl = (1, 1, 1, view)
	if color == 2:
		cl = (0, 1, 1, view)
	if color == 3:
		cl = (1, 0, 0, view)

	ax.add_collection3d(Poly3DCollection(verts, linewidths=1, facecolors = cl, edgecolors='k', alpha=.25))

	ax.set_aspect('equal')

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	plt.show()

plot_cube(int(input()), int(input()), float(input()))
