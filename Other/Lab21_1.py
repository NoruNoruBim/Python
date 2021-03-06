import requests
from bs4 import BeautifulSoup

url_end = ''
print('Write your request:')
url_end = input()

def make_url(url_end):

    url_base = 'https://ru.wikipedia.org/wiki/'

    url = url_base + url_end

    return url


def get_request(url_end):

    request = requests.get(make_url(url_end))

    with open('request.txt', 'w', encoding='utf8') as file:
        file.write(request.text)

    return request.text


def cleaning(url_end):

    soup = BeautifulSoup((get_request(url_end)), "lxml")

    main_information = soup.find('div', {'id' : 'bodyContent'})

    with open('request_clean.txt', 'w', encoding='utf8') as file:
        file.write(main_information.text)

    return main_information.text

print('\nSuccess!\nYou can open file  request_clean.txt')
print('\nYou can also print information on console\nWrite yes or no:')

choice = input()
if choice == 'yes':
    print(cleaning(url_end))
    print('\nSee you later!')
else:
    cleaning(url_end)
    print('\nOk, see you later!')
