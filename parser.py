import requests
from bs4 import BeautifulSoup

url = "https://www.rudn.ru/contacts"
   
r = requests.get(url)
#print(r)

html = BeautifulSoup(r.text, "html.parser")
#print(html)

p = html.find_all("div", class_="contacts__item phone")

print(p)

for i in p:
    print(i.find("a").text.strip())
