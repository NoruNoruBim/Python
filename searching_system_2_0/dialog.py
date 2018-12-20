import crawler
import indexator
import calc_expr


def start():
	noa = crawler.make_base()
	index = indexator.indexation(noa)
	requests = []
	output = []

	with open("requests.txt", 'r', encoding="utf8") as file:# достаем запросы из файла
		for line in file:
			requests += [line]

	for request in requests:#			обрабатываем каждый запрос
		tmp = calc_expr.calc(request, index, noa)#	итог, но в бинарном виде
		tmp2 = []
		for i in range(len(tmp)):#				переводим в пользовательский вид
			if tmp[i]: tmp2 += [str(i + 1) + ".txt"]
		output += [tmp2]
	
	# печатаем итоговый итог
	print("\nOutput:")
	for i in range(len(requests)):
		print("req: " + requests[i].strip() + " --- output: " + str(output[i]))


def main():
	print("Enter mode. 1 - batch processing, 2 - manual processing.")
	tmp = input()

	if tmp == '2':
		articles = []
		print("Enter articles to search:")
		while tmp != '0':
			tmp = input()
			articles += tmp.split()
			with open("articles.txt", 'w') as fi:
				for i in range(len(articles) - 1):
					fi.write(articles[i] + '\n')
		print("Enter your request:")
		tmp = input()
		with open("requests.txt", 'w') as req:
			req.write(tmp + '\n')
	start()

main()

