'''
Программа аппроксимации до прямого усеченного конуса
Фигуры из лр №2, используя функции из лр №3 и библиотеку OpenGl
Выполнил студент гр. М8о-307 Баранов А.А.
'''

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys
import numpy as np
from math import sqrt

# Объявляем все глобальные переменные
global xrot         # Величина вращения по оси x
global yrot         # Величина вращения по оси y
global ambient      # рассеянное освещение
global cubecolor    # Цвет куба
global lightpos     # Положение источника освещения
global var			# Вариации построения, определяемые пользователем


# Процедура инициализации
def init():
	global xrot         # Величина вращения по оси x
	global yrot         # Величина вращения по оси y
	global ambient      # Рассеянное освещение
	global cubecolor    # Цвет куба
	global lightpos     # Положение источника освещения
	global var

	var = 0
	xrot = 0.0                          # Величина вращения по оси x = 0
	yrot = 0.0                          # Величина вращения по оси y = 0
	ambient = (1.0, 1.0, 1.0, 1)        # Первые три числа цвет в формате RGB, а последнее - яркость (1 1 1 - white)
	cubecolor = (0, 0, 0, 1)            # аквамарин
	lightpos = (1.0, 1.0, 1.0)          # Положение источника освещения по осям xyz

	glClearColor(0.5, 0.5, 0.5, 1.0)                # Серый цвет для первоначальной закраски
	gluOrtho2D(-4.0, 4.0, -4.0, 4.0)                # Определяем границы рисования по горизонтали и вертикали (отдаление от рисунка при открытии окна)
	glRotatef(-90, 1.0, 0.0, 0.0)                   # Сместимся по оси Х на 90 градусов (положение рисунка при открытии окна)
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient) # Определяем текущую модель освещения
	glEnable(GL_LIGHTING)                           # Включаем освещение
	glEnable(GL_LIGHT0)                             # Включаем один источник света
	glLightfv(GL_LIGHT0, GL_POSITION, lightpos)     # Определяем положение источника света

# Процедура обработки специальных клавиш
def specialkeys(key, x, y):
	global xrot
	global yrot
	global var
	# Обработчики для клавиш со стрелками
	if key == GLUT_KEY_UP:      # Клавиша вверх
		xrot -= 5.0             # Уменьшаем угол вращения по оси Х
	if key == GLUT_KEY_DOWN:    # Клавиша вниз
		xrot += 5.0             # Увеличиваем угол вращения по оси Х
	if key == GLUT_KEY_LEFT:    # Клавиша влево
		yrot -= 5.0             # Уменьшаем угол вращения по оси Y
	if key == GLUT_KEY_RIGHT:   # Клавиша вправо
		yrot += 5.0             # Увеличиваем угол вращения по оси Y
	if key == GLUT_KEY_F1:
		if var == 1:
			var = 2
		elif var == 2:
			var = 3
		else:
			var = 1
	if key == GLUT_KEY_F2:# точность аппроксимации
		var = 4
	if key == GLUT_KEY_F3:# освещение
		var = 5


	glutPostRedisplay()         # Вызываем процедуру перерисовки

def choose_color(color):
	if color == 1:
		cl = (0, 0, 0, 1)# black
	if color == 2:
		cl = (1, 1, 1, 1)# white
	if color == 3:
		cl = (0, 1, 1, 1)# aquamarine
	if color == 4:
		cl = (1, 0, 0, 1)# red
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
		list_of_points += [list_of_points[i - 1] * percent]# cube on new level is less than old cubes
		list_of_points[i][:, 2] /= percent
		list_of_points[i][:, 2] += old_height

	for i in range(len(list_of_points)):# now we have "cake" with some levels of quadratic figures, we need to circle it.
		circle_it(list_of_points, i, alpha)
	for i in range(len(list_of_points)):# finish. add all cubes (in summary OUR CONE) to the final list.
		cubes += [make_cube_sides(list_of_points[i])]

	return cubes



def draw():
	global xrot
	global yrot
	global lightpos
	global cubecolor
	global var
	
	R = 4.
	r = 2.
	height = 10.
	levels = 10
	alpha = 10
	
	'''
	light2_diffuse = [0.4, 0.7, 0.2];
	light2_position = [0.0, 0.0, 1.0, 1.0];
	glEnable(GL_LIGHT2);
	glLightfv(GL_LIGHT2, GL_DIFFUSE, light2_diffuse);
	glLightfv(GL_LIGHT2, GL_POSITION, light2_position);
	glLightf(GL_LIGHT2, GL_CONSTANT_ATTENUATION, 0.0);
	glLightf(GL_LIGHT2, GL_LINEAR_ATTENUATION, 0.2);
	glLightf(GL_LIGHT2, GL_QUADRATIC_ATTENUATION, 0.4);
	
	light3_diffuse = [0.4, 0.7, 0.2];
	light3_position = [0.0, 0.0, 1.0, 1.0];
	light3_spot_direction = [0.0, 0.0, -1.0];
	glEnable(GL_LIGHT3);
	glLightfv(GL_LIGHT3, GL_DIFFUSE, light3_diffuse);
	glLightfv(GL_LIGHT3, GL_POSITION, light3_position);
	glLightf(GL_LIGHT3, GL_SPOT_CUTOFF, 30);
	glLightfv(GL_LIGHT3, GL_SPOT_DIRECTION, light3_spot_direction);'''

	if var == 1:
		cubecolor = (1, 1, 1, 1)
	if var == 2:
		cubecolor = (0, 1, 1, 1)
	if var == 3:
		cubecolor = (1, 0, 0, 1)
	if var == 4:
		levels = 15
		alpha = 20
	if var == 5:
		glEnable(GL_LIGHT0)
		light0_diffuse = [1.0, 1.0, 1.0];
		light0_direction = [1.0, 1.0, 1.0, 1];
		glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
		glLightfv(GL_LIGHT0, GL_POSITION, light0_direction);


	glClear(GL_COLOR_BUFFER_BIT)                                # Очищаем экран и заливаем серым цветом
	glPushMatrix()                                              # Сохраняем текущее положение "камеры"
	glRotatef(xrot, 1.0, 0.0, 0.0)                              # Вращаем по оси X на величину xrot
	glRotatef(yrot, 0.0, 1.0, 0.0)                              # Вращаем по оси Y на величину yrot
	if var != 5:
		glLightfv(GL_LIGHT0, GL_POSITION, lightpos)                 # Источник света вращаем вместе с фигурой

	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, cubecolor)

	p = np.array([[-1., -1., -1.],# points in 3d Cartesian coordinates
				  [1., -1., -1.],
				  [1., 1., -1.],
				  [-1., 1., -1.],
				  [-1., -1., 1.],
				  [1., -1., 1.],
				  [1., 1., 1.],
				  [-1., 1., 1.]])

	R *= 0.2
	r *= 0.2
	height *= 0.2
	
	param = 0.5 * R * sqrt(2)# we wanna in the function "circle_it" make CIRCRUMSCRIBED circle (описанную) a = R * 2^0.5
	height -= R * sqrt(2) * 0.25# cube height not from 0! -> from -1!
	p[:, 2] *= (height / levels) / param# shrink
	p *= param
	
	cubes = approximate(p, levels, height, R, r, alpha)
	
	# plotting figure
	for q in range(len(cubes)):
		for i in range(len(cubes[q])):
			glBegin(GL_QUADS)
			glColor3f(0., 0., 0.)
			for j in range(len(cubes[q][i])):
				#print(str(i) + " " + str(j))
				x = cubes[q][i][j][0]
				y = cubes[q][i][j][1]
				z = cubes[q][i][j][2]
				glVertex3f(x, y, z)
			glEnd()

	glPopMatrix()                                               # Возвращаем сохраненное положение "камеры"
	glutSwapBuffers()                                           # Выводим все нарисованное в памяти на экран


# Здесь начинается выполнение программы
# Использовать двойную буферизацию и цвета в формате RGB (Красный, Зеленый, Синий)
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
# Указываем начальный размер окна (ширина, высота)
glutInitWindowSize(480, 480)
# Указываем начальное положение окна относительно левого верхнего угла экрана
glutInitWindowPosition(50, 50)
# Инициализация OpenGl
glutInit(sys.argv)
# Создаем окно с заголовком "Cube"
glutCreateWindow(b"Cube")
# Определяем процедуру, отвечающую за перерисовку
glutDisplayFunc(draw)
# Определяем процедуру, отвечающую за обработку клавиш
glutSpecialFunc(specialkeys)
# Вызываем нашу функцию инициализации
init()
# Запускаем основной цикл
glutMainLoop()
