import bs4
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as soup
import nltk
from newspaper import Article
import csv

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'}

reg_url = "https://www.infowars.com/category/14/"
nltk.download("punkt")
# Opening the connection and grabbing the page
req = Request(url=reg_url, headers=headers)
html = urlopen(req).read()

# req.close()

# Html parsing
page_soup = soup(html, 'html.parser')

containers = page_soup.find_all(
    "a", attrs={"class": "css-1xjmleq"})

data_list = [["Title", "Author", "Text", "Label"]]


with open('InfowarsHealth.csv', 'w', newline='', encoding="utf-8") as file:
    fieldnames = ['Title', 'Author', 'Text', 'Label']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for item in containers:
        reference = item["href"]
        print(reference)

        news_url = "https://www.infowars.com"+reference
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
