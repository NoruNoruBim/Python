from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np

global c


def Lagrange(x):
	global c

	s = 0.0
	for j in range(len(c)):
		ph = 1.0
		pl = 1.0
		for i in range(len(c)):
			if i == j: continue
			ph *= x - c[i][0]
			pl *= c[j][0] - c[i][0]
		s += c[j][1] * ph / pl
	return s
 
def Initialize():
	glClearColor(1.0, 1.0, 1.0, 1.0)
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	gluOrtho2D(-5, 5, -5, 5)
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
 
def Draw():
	global c

	glClear(GL_COLOR_BUFFER_BIT)
	
	glColor3f(0.0, 0.0, 0.0)# black
	glBegin(GL_LINES)# 			координатные оси
	glVertex2d(-5., 0.)
	glVertex2d(5., 0.)
	glVertex2d(0., -5.)
	glVertex2d(0., 5.)
	glVertex2d(5., 0.)
	glVertex2d(4.7, 0.2)
	glVertex2d(5., 0.)
	glVertex2d(4.7, -0.2)
	glVertex2d(0., 5.)
	glVertex2d(-0.1, 4.7)
	glVertex2d(0., 5.)
	glVertex2d(0.1, 4.7)
	glEnd()
 
	glColor3f(1., 0., 0.)
	glPointSize(2.0)
	glBegin(GL_POINTS)
	for x in np.arange(-1.0, 1.0, 0.001): 
		glVertex2d(x, Lagrange(x))
	glEnd()
	
	
	glColor3f(0., 0., 1.)
	glPointSize(5.0)
	glBegin(GL_POINTS)
	for i in range(len(c)):
		glVertex2d(c[i][0], c[i][1])
	glEnd()

	glFlush()

def main():
	global c

	print("Choose mod. 1 - auto, 2 - manual.")
	tmp = input()
	if tmp == '1':
		c = [(-1., 0.5), (-0.6, 0.8), (-0.2, 1.0), (0.5, 0.1), (1.0, 0.9)]# our points
	else:
		print("Enter 5 points:")
		c = [[0., 0.], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
		point = np.array(2)
		for i in range(5):
			c[i][0] = float(input())
			c[i][1] = float(input())

	glutInit()
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
	glutInitWindowSize(800, 500)
	glutCreateWindow("Lagrange")
	glutDisplayFunc(Draw)
	Initialize()
	glutMainLoop()


main()


