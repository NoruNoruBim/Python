import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter# special dict


def get_bases():
	#df1 = pd.read_csv("C:/hello_world/Python/AI/databases/online_shoppers_intention.csv", header=None)
	#print("Dataframe1 done.")

	df2 = []
	with open("C:/hello_world/Python/AI/databases/SMSSpamCollection", 'r', encoding='utf8') as file:
		for line in file:
			tmp = line.strip().split()
			df2 += [[[tmp[0]], tmp[1:]]]
	print("Dataframe2 (sms) done.")

	return df2

def clear(word):
	while not word[0].isalnum() and len(word) > 1:
		word = word[1:]
	while not word[-1].isalnum() and len(word) > 1:
		word = word[:-1]
	return word
	
def get_filtered_dict(d):
	x1 = list(filter(lambda x: len(x) > 3 and x.isalnum(), d.keys()))
	y1 = [d[i] for i in x1]

	d.clear()
	for i in range(len(x1)):
		d[x1[i]] = y1[i]

def plot_diagrams(n, stat):
	print(np.array(stat.most_common(30)))
	
	x = list(np.array(stat.most_common(n)).T[0])
	y = list(map(int, np.array(stat.most_common(n)).T[1]))

	fig = plt.figure()
	plt.bar(x, y)
	plt.title("Распределение слов")
	plt.grid(True)   # линии вспомогательной сетки

	plt.show()


def main():
	df2 = np.array(get_bases())
	stat = Counter()# special dict

	for sms in df2.T[1]:
		for word in sms:
			word = clear(word.lower())
			stat[word] += 1
	plot_diagrams(30, stat)

	get_filtered_dict(stat)
	plot_diagrams(30, stat)

main()
