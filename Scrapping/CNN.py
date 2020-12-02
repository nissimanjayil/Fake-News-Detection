import newspaper
from newspaper import Article
import csv
import nltk

# prep articles from url
cnn_paper = newspaper.build('https://edition.cnn.com/',language='en')

nltk.download('punkt')
news = []

# for every section collect articles urls
for category in cnn_paper.category_urls():
    cnn_paper = newspaper.build(category)
    for article in cnn_paper.articles:
        news.append(article)
        
        
data_list = [["Title", "Author", "Text", "Label"]]

# store articles
if len(news) > 0:

    with open('CNN.csv', 'w', newline='', encoding="utf-8") as file:
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