import bs4
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import nltk
from newspaper import Article
import csv

url = "https://www.breitbart.com/"
nltk.download("punkt")
# Opening the connection and grabbing the page
client = uReq(url)
raw_html = client.read()

client.close()

# Html parsing
page_soup = soup(raw_html, 'html.parser')

containers = page_soup.findAll("section", {"class": "gb"})

data_list = [["Title", "Author", "Text", "Label"]]

with open('falsenews.csv', 'w', newline='', encoding="utf-8") as file:
    fieldnames = ['Title', 'Author', 'Text', 'Label']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for container in containers:
        reference = container.article.footer.h2.a["href"]
        news_url = "https://www.breitbart.com"+reference
        article = Article(news_url)
        article.download()
        article.parse()
        article.nlp()
        title = article.title
        Author = article.authors
        Text = article.text
        Label = -1
        writer.writerow({'Title': title, 'Author': Author,
                         'Text': Text, 'Label': Label})
