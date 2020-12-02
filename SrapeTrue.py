import bs4
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import nltk
import requests
from newspaper import Article
import csv

url = 'https://www.bbc.com/news'

nltk.download("punkt")
# Opening the connection and grabbing the page
client = uReq(url)
raw_html = client.read()

client.close()

# HTML parsing
page = soup(raw_html,'html.parser')
contents =[]

# Main headline articles
contents.append(page.find('div',{'class':'gel-layout__item nw-c-top-stories__secondary-item gel-1/1 gel-1/3@m gel-1/4@l nw-o-keyline nw-o-no-keyline@m gs-u-float-left nw-c-top-stories__secondary-item--1 gel-3/16@xxl gs-u-float-none@xxl gs-u-mt gs-u-mt0@xs'}))
for i in range(2,5):
    contents.append(page.find('div', {'class':f'gel-layout__item nw-c-top-stories__secondary-item gel-1/1 gel-1/3@m gel-1/4@l nw-o-keyline nw-o-no-keyline@m gs-u-float-left nw-c-top-stories__secondary-item--{i} gel-1/5@xxl'}))
    
# Top bar articles
contents.append(page.find('div',{'class':'nw-c-top-stories__tertiary-top gel-2/3@m gel-3/4@l gel-1/5@xxl gs-u-float-right@m'}))
for i in range(1,4):
    contents.append(page.find('div', {'class':f'gel-layout__item nw-c-top-stories__tertiary-items gel-1/2@m gel-1/3@l gel-1/1@xxl nw-o-keyline nw-o-no-keyline@m nw-c-top-stories__tertiary-top-item--{i}'}))

# Bottom bar articles
contents.append(page.find('div',{'class':'nw-c-top-stories__tertiary-bottom gel-2/3@m gel-3/4@l gel-1/1@xxl gs-u-float-right@m gs-u-float-none@xxl'}))
for i in range(2,5):
    contents.append(page.find('div',{'class':f'gel-layout__item nw-c-top-stories__tertiary-items gel-1/2@m gel-1/3@l gel-1/5@xxl nw-o-keyline nw-o-no-keyline@m nw-c-top-stories__tertiary-bottom-item--{i}'}))

data_list = [["Title", "Author", "Text", "Label"]]

with open('truenews.csv', 'w', newline='', encoding="utf-8") as file:
    fieldnames = ['Title', 'Author', 'Text', 'Label']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for content in contents:
        reference = content.a["href"]
        news_url = "https://www.bbc.com"+reference
        article = Article(news_url)
        article.download()
        article.parse()
        article.nlp()
        title = article.title
        Author = article.authors
        Text = article.text
        Label = 1
        writer.writerow({'Title': title, 'Author': Author,
                         'Text': Text, 'Label': Label})


#link = search_results[2].a["href"]
'''for x in search_results:
    link = x.a["href"]
    print(link)'''

