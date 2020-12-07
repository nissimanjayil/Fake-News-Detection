import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import nltk
import re
import string


stopwords = nltk.corpus.stopwords.words('english')


class FakeNewsDetector(object):

    '''
        This function cleans the data
        1)Removes the punctuations
        2)tokenizes the sentences
        3)Removes stopwords
    '''

    def clean_data(self, dataset):
        # Checks if the dataset contains null values
        dataset = dataset.fillna(' ')
        # Creates a new column with entire Article containing the title,author and the text
        dataset['Article'] = dataset['Title'] + \
            ' '+dataset['Author']+' '+dataset['Text']

        # Iterate through each article and remove any punctuations,tokenises and remove any stopwords
        for index, row in dataset.iterrows():
            article = row['Article']
            # REMOVE PUNCTUATIONS
            remove_punctuations = "".join(
                [char for char in article if char not in string.punctuation])
            # TOKENISATION
            tokenise = re.split('\W+', remove_punctuations)
            # REMOVE STOPWORDS
            filtered_text = (" ").join(word.lower()
                                       for word in tokenise if word not in stopwords)
            # UPDATE THE ROW
            dataset.loc[index, 'Article'] = filtered_text
        return dataset

    '''


    '''

    def vectorise_text(self, data):
        # encodes data into a numerical value
        cv = CountVectorizer()
        # Counts the number of frequencies of words in text and return a vector
        cv.fit_transform(data)
        count_matrix = cv.transform(data)
        tfidf = TfidfTransformer(norm='l2')
        tfidf.fit(count_matrix)
        tfidf_matrix = tfidf.fit_transform(count_matrix)
        # print(tfidf_matrix.toarray().shape)
        return tfidf_matrix


def main():
    Fake_Detector = FakeNewsDetector()
    df = pd.read_csv("sample.csv")
    cleaned_dataset = Fake_Detector.clean_data(df)
    # print(cleaned_dataset.head())
    dataset = cleaned_dataset[['Article', 'Label']]
    X = dataset.iloc[:, 0]
    Y = dataset.iloc[:, 1]

    vectoriser = Fake_Detector.vectorise_text(X)


if __name__ == '__main__':
    main()
