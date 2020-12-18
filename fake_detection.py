import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
import nltk
import re
import string
import seaborn as sns

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


def logistic_Regression(X_train, X_test, Y_train, Y_test):
    pipe1 = Pipeline([('vect', CountVectorizer()), ('tfidf',
                                                    TfidfTransformer()), ('model', LogisticRegression())])

    model_lr = pipe1.fit(X_train, Y_train)
    lr_pred = model_lr.predict(X_test)
    print(lr_pred)


def knn_classifier(X_train, X_test, Y_train, Y_test):

    pipe1 = Pipeline([('vect', CountVectorizer()), ('tfidf',
                                                    TfidfTransformer()), ('model', KNeighborsClassifier(n_neighbors=3, weights='uniform'))])

    model_knn = pipe1.fit(X_train, Y_train)
    knn_pred = model_knn.predict(X_test)
    print(knn_pred)


def main():
    Fake_Detector = FakeNewsDetector()
    df = pd.read_csv("dataset.csv")
    cleaned_dataset = Fake_Detector.clean_data(df)

    dataset = cleaned_dataset[['Article', 'Resource', 'Label']]
    # print(dataset.head())
    X = dataset.iloc[:, 0]
    Z = dataset.iloc[:, 1]
    Y = dataset.iloc[:, 2]
    # sns.countplot(dataset['Label'])
    # plt.show()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=1)

    logistic_Regression(X_train, X_test, Y_train, Y_test)

    knn_classifier(X_train, X_test, Y_train, Y_test)


if __name__ == '__main__':
    main()
