from pythonds.basic.stack import Stack
#import operator
import numpy as np


def arr(word, index, noa):
	tmp2 = np.zeros(noa)
	if word in index.keys():
		tmp1 = index[word]
		for i in tmp1:#				vector with ones and zeros
			tmp2[i - 1] = 1
	return tmp2

def fix(line):
	count = 0
	for char in line:
		count += 1
		if char == '(':
			line = line[:count] + ' ' + line[count:]
			count += 1
		if char == ')':
			line = line[:count - 1] + ' ' + line[count - 1:]
			count += 1
	return line

def infixToPostfix(infixexpr):
	infixexpr = fix(infixexpr)
	prec = {}
	#prec['*'] = 3
	prec['~'] = 4
	prec['&'] = 3
	#prec['/'] = 3
	#prec['+'] = 2
	prec['|'] = 2
	#prec['-'] = 2
	prec['('] = 1
	opStack = Stack()
	postfixList = []
	tokenList = infixexpr.split()

	for token in tokenList:
		#if token in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or token in "0123456789":
		if token.isalnum():
			postfixList.append(token)
		elif token[0] == '(':
			opStack.push(token)
		elif token == ')':
			topToken = opStack.pop()
			while topToken != '(':
				postfixList.append(topToken)
				topToken = opStack.pop()
		else:
			while (not opStack.isEmpty()) and \
			   (prec[opStack.peek()] >= prec[token]):
				  postfixList.append(opStack.pop())
			opStack.push(token)

	while not opStack.isEmpty():
		postfixList.append(opStack.pop())
	return " ".join(postfixList)

'''
print(infixToPostfix("A * B + C * D"))
print(infixToPostfix("(A | B) & C | (D | E) & (F | G)"))'''

def log_mul(a, b):
	return a * b

def log_sum(a, b):
	for i in range(len(a)):
		if a[i] + b[i] >= 1: a[i] = 1
	return a

def log_not(a):
	for i in range(len(a)):
		if a[i] == 1: a[i] = 0
		else: a[i] = 1

def calc(expr, index, noa):
	expr = infixToPostfix(expr)
	#OPERATORS = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
	OPERATORS = {'|': log_sum, '&': log_mul, '~': log_not}
	stack = []
	for token in expr.split(" "):
		if token in OPERATORS:
			op2, op1 = stack.pop(), stack.pop()
			stack.append(OPERATORS[token](op1, op2))
		else:
			stack.append(arr(token, index, noa))
	return stack.pop()

