import newspaper
from newspaper import Article
import csv
import nltk

bbc_paper = newspaper.build('https://www.bbc.com/news',language='en')

nltk.download('punkt')
news = []

# for each section collect urls for articles
for category in bbc_paper.category_urls():
    bbc_paper = newspaper.build(category)
    for article in bbc_paper.articles:
        news.append(article)
        
        
data_list = [["Title", "Author", "Text", "Label"]]

# collect articles
if len(news) > 0:

    with open('BBC.csv', 'w', newline='', encoding="utf-8") as file:
        fieldnames = ['Title', 'Author', 'Text', 'Label']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for article in news:
            try:
                article.download()
                article.parse()
                article.nlp()

                title = article.title
                author = []
                text = article.text
                label =1
                writer.writerow({'Title':title,'Author': author,'Text': text, 'Label': label })
            except:
                print("failed")