import bs4
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import nltk
import requests
from newspaper import Article
import csv

url = 'https://edition.cnn.com/search?q=coronavirus&sort=newest&category=business,us,politics,world,opinion,health&size=100'
nltk.download("punkt")
# Opening the connection and grabbing the page
client = uReq(url)
raw_html = client.read()

#client.close()

page = soup(raw_html,'html.parser')
search_results = page.find('div', {'class':'cnn-search__results-list'})
contents = search_results.findAll('div', {'class':'cnn-search__result-contents'})
for content in contents:
    link = content.h3.a["href"]
    print(link)