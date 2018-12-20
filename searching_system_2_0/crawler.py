import requests
from bs4 import BeautifulSoup
import os, shutil

def make_base():
	print("--- CRAWLER ---")
	url_array = []
	urls_new = set()
	with open("articles.txt", 'r') as url_s:# make array of articles url
		for url in url_s:
			url_array += ["https://en.wikipedia.org/wiki/" + url.strip()]

	shutil.rmtree("data", True)
	os.mkdir("data")

	print("--- making base ---")
	for i in range(len(url_array)):
		print("--- file №" + str(i + 1) + " ---")
		request = requests.get(url_array[i])# реквестируем конкретную ссылку

		soup = BeautifulSoup((request.text), "lxml")# преобразуем в текст

		main_information = soup.find('div', {'id' : 'bodyContent'})# выискиваем то что нужно

		filename = str(i + 1) + '.txt'
		with open("data/" + filename, 'w', encoding='utf8') as file:# записываем выгруженную с википедии статью в нашу базу
			file.write(main_information.text)
	
	return len(url_array)


#make_base()
