import bs4
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import nltk
from newspaper import Article
import csv


url = "https://www.theonion.com/tag/opinion"
nltk.download("punkt")
# Opening the connection and grabbing the page

client = uReq(url)
raw_html = client.read()

client.close()

# Html parsing
page_soup = soup(raw_html, 'html.parser')

containers = page_soup.findAll(
    "div", {"class": "cw4lnv-5 aoiLP"})


data_list = [["Title", "Author", "Text", "Label"]]

with open('Onionopinion.csv', 'w', newline='', encoding="utf-8") as file:
    fieldnames = ['Title', 'Author', 'Text', 'Label']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for item in containers:
        reference = item.a["href"]
        print(reference)

        news_url = reference
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
