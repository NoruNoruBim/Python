'''
Программа аппроксимации до прямого усеченного конуса
Фигуры из лр №2
Выполнил студент гр. М8о-307 Баранов А.А.
'''

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def choose_color(color, view):
	if color == 1:
		cl = (0, 0, 0, view)# black
	if color == 2:
		cl = (1, 1, 1, view)# white
	if color == 3:
		cl = (0, 1, 1, view)# aquamarine
	if color == 4:
		cl = (1, 0, 0, view)# red
	return cl
	
def make_cube_sides(p):
	sides = [[p[0],p[1],p[2],p[3]],# bottom side
			 [p[4],p[5],p[6],p[7]],# top side
			 [p[0],p[1],p[5],p[4]],# next other sides
			 [p[1],p[2],p[6],p[5]],# (p[i] DEPENDS on points locates)
			 [p[2],p[3],p[7],p[6]],
			 [p[3],p[0],p[4],p[7]]]
	return sides

def circle_it(list, ind, n):
	points = []
	# logical we "rotate" cube, but IRL we add new rotated cubes to the group
	# to rotate cube, we need to change only X and Y (Z is const in this function and equal with SAMPLE, size too)
	c = sqrt(((list[ind][0][0] - list[ind][1][0]) ** 2 + (list[ind][0][1] - list[ind][1][1]) ** 2 + (list[ind][0][2] - list[ind][1][2]) ** 2) * 2.) / 2.# size of side (?)
	z_2 = list[ind][4][2]
	z_1 = list[ind][0][2]

	for i in range(n):# n - number of added cubes
		x_1 = -c + (c * 2 / n) * i# turn (?)
		y_1 = sqrt(c ** 2 + 0.000000000000002 - x_1 ** 2)
		
		y_2 = c - (c * 2 / n) * i# simmetric with prev
		x_2 = sqrt(c ** 2 + 0.000000000000002 - y_2 ** 2)
		
		points += [[x_1, y_1, z_1]] + [[x_2, y_2, z_1]] + [[-x_1, -y_1, z_1]] + [[-x_2, -y_2, z_1]]
		points += [[x_1, y_1, z_2]] + [[x_2, y_2, z_2]] + [[-x_1, -y_1, z_2]] + [[-x_2, -y_2, z_2]]
		list += [points]
		points = []
	return list

def approximate(p, l, h, R, r, alpha):
	list_of_points = [p]# list of different cubes points (points stores in the groupes)
	cubes = []
	if R / r > 4.:# crutch
		r = R / 4.
	percent = pow(r/R, 1./l)
	old_height = h / l
	for i in range(1, l):# make "cake" from a lot of cubes. some levels and roundness.
		'''
		list_of_points += [list_of_points[i - 1] * 0.95]# cube on new level is less than old cubes
		list_of_points[i][:, 2] += ((0.95 + i * 0.0025) ** i) * beta# +- params to good visualization. "+i*0.0025" needs to straight, not circle edges.'''
		list_of_points += [list_of_points[i - 1] * percent]# cube on new level is less than old cubes
		list_of_points[i][:, 2] /= percent
		list_of_points[i][:, 2] += old_height

	for i in range(len(list_of_points)):# now we have "cake" with some levels of quadratic figures, we need to circle it.
		circle_it(list_of_points, i, alpha)
	for i in range(len(list_of_points)):# finish. add all cubes (in summary OUR CONE) to the final list.
		cubes += [make_cube_sides(list_of_points[i])]

	return cubes

def plot_cube(param=1., color=3, view=0.2, lw_par=0.1, mode=1, height=5., R=4., r=1., levels=10, alpha=20):
	# beginning:
	points = np.array([[-1., -1., -1.],# points in 3d Cartesian coordinates
					   [1., -1., -1.],
					   [1., 1., -1.],
					   [-1., 1., -1.],
					   [-1., -1., 1.],
					   [1., -1., 1.],
					   [1., 1., 1.],
					   [-1., 1., 1.]])
	# basicly |side| = 2
	if mode == 2:
		param = 0.5 * R * sqrt(2)# we wanna in the function "circle_it" make CIRCRUMSCRIBED circle (описанную) a = R * 2^0.5
		height -= R * sqrt(2) * 0.25# cube height not from 0! -> from -1!
		points[:, 2] *= (height / levels) / param# shrink
	points *= param

	# plotting:
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")# 111 means 1x1 grid, first subplot

	if mode == 1:# simple mode
		cubes = [make_cube_sides(points)]
	elif mode == 2:# approximation mode
		cubes = approximate(points, levels, height, R, r, alpha)

	# plot cubes
	for i in range(len(cubes)):
		ax.add_collection3d(Poly3DCollection(cubes[i], linewidths=lw_par, facecolors = choose_color(color, view), edgecolors='k'))

	# customize plot
	ax.set_aspect("equal")
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	if mode == 2:
		limit = height * 1.2
	else:
		limit = param * 2
	plt.xlim([-limit, limit])
	plt.ylim([-limit, limit])
	ax.set_zlim(-limit, limit)

	plt.show()

def menu():
	print("Initialization...")
	if input() == 'a':# full auto to fast debugging
		plot_cube(1, 3, 0.1, 0.2, 2, 15., 8., 1, 20, 10)
		#		  p  c   v    lw  m   h   R    r    l   a
		return
	print("Enter size param.")
	sz_par = float(input())
	print("Enter color. 1 - black, 2 - white, 3 - aquamarine, 4 - red.")
	color = int(input())
	print("Enter saturation parameter (in gradation from 0.1 to 1).")# насыщенность
	sat_par = float(input())
	print("Enter line widths (in gradation from 0 to 1).")
	lw_par = float(input())
	print("Do you want make just cube or approximated cone? 1 - cube, 2 - cone.")
	tmp = int(input())

	if tmp == 1:
		plot_cube(sz_par, color, sat_par, lw_par, 1)# just cube
	elif tmp == 2:
		print("Enter height of cone.")
		height = int(input())
		print("Enter biggest radius of cone.")
		R = float(input())
		print("Enter smallest radius of cone.")
		r = float(input())
		print("Enter accuracy of roundness.")# закругленность
		alpha = int(input())
		print("Enter number of levels.")
		levels = int(input())
		print("plotting...")

		plot_cube(sz_par, color, sat_par, lw_par, 2, height, R, r, levels, alpha)# cube with approximation to cone
	print("-- end of work --")
	return

menu()
