import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def plot_cube():
	points = np.array([[-1, -1, -1],
					  [1, -1, -1 ],
					  [1, 1, -1],
					  [-1, 1, -1],
					  [-1, -1, 1],
					  [1, -1, 1 ],
					  [1, 1, 1],
					  [-1, 1, 1]])

	Z = np.zeros((8,3))
	for i in range(8): Z[i,:] = points[i,:]
	Z = 10.0 * Z

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	r = [-1,1]

	X, Y = np.meshgrid(r, r)
	# plot vertices
	ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

	# list of sides' polygons of figure
	verts = [[Z[0],Z[1],Z[2],Z[3]],
	 [Z[4],Z[5],Z[6],Z[7]], 
	 [Z[0],Z[1],Z[5],Z[4]], 
	 [Z[2],Z[3],Z[7],Z[6]], 
	 [Z[1],Z[2],Z[6],Z[5]],
	 [Z[4],Z[7],Z[3],Z[0]], 
	 [Z[2],Z[3],Z[7],Z[6]]]

	# plot sides
	ax.add_collection3d(Poly3DCollection(verts, linewidths=1, facecolors = (0,1,1,1), edgecolors='k', alpha=.25))

	ax.set_aspect('equal')

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	plt.show()

plot_cube()
