import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_curve, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
import nltk
import re
import string
import seaborn as sns

# nltk.download('stopwords')
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


def baseline_model(x, y, z):
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(x, y)
    dummy_pred = dummy.predict(x)
    data = {'resource': z,
            'label': dummy_pred}
    df = pd.DataFrame(data)
    sns.countplot(x='label', hue='resource',
                  data=df, palette='deep')
    plt.ylabel('No of Articles')
    plt.xlabel('Label')
    plt.title("Baseline Model")
    plt.show()


def logistic_Regression(X_train, X_test, Y_train, Y_test, Z_train, Z_test):
    pipe1 = Pipeline([('vect', CountVectorizer()), ('tfidf',
                                                    TfidfTransformer()), ('model', LogisticRegression(solver='lbfgs', C=10))])

    model_lr = pipe1.fit(X_train, Y_train)
    lr_pred = model_lr.predict(X_test)
    data = {'resource': Z_test,
            'label': lr_pred}
    df = pd.DataFrame(data)
    sns.countplot(x='label', hue='resource',
                  data=df, palette='deep')
    plt.ylabel('No of Articles')
    plt.xlabel('Label')
    plt.title("Logistic Regression predictions")
    plt.show()


def logistic_Regression_crossval(x, y, z):
    kf = KFold(n_splits=5)
    mean_error = []
    std_error = []
    c_range = [0.01, 1, 10, 100]

    for c in c_range:

        pipe1 = Pipeline([('vect', CountVectorizer()), ('tfidf',
                                                        TfidfTransformer()), ('model', LogisticRegression(solver='lbfgs', penalty='l2', C=c, random_state=0))])
        tmp = []
        plotted = False
        for train, test in kf.split(x):
            model_lr = pipe1.fit(x[train], y[train])
            lr_pred = model_lr.predict(x[test])
            tmp.append(mean_squared_error(y[test], lr_pred))

        mean_error.append(np.array(tmp).mean())
        std_error.append(np.array(tmp).std())

    plt.errorbar(c_range, mean_error, yerr=std_error)
    plt.xlabel('C value')
    plt.ylabel('Mean Square Error')
    plt.title(" C value for Logistic Regression performance")
    plt.show()


def knn_classifier(X_train, X_test, Y_train, Y_test, Z_train, Z_test):
    pipe1 = Pipeline([('vect', CountVectorizer()), ('tfidf',
                                                    TfidfTransformer()), ('model', KNeighborsClassifier(n_neighbors=3, weights='uniform'))])

    model_knn = pipe1.fit(X_train, Y_train)
    knn_pred = model_knn.predict(X_test)
    data = {'resource': Z_test,
            'label': knn_pred}
    df = pd.DataFrame(data)
    sns.countplot(x='label', hue='resource',
                  data=df, palette='deep')
    plt.title("KNN Classifier Predictions")
    plt.xlabel("Label")
    plt.ylabel("No. of Articles")
    plt.show()


def knn_classifier_crossval(x, y, z):
    kf = KFold(n_splits=5)
    mean_error = []
    std_error = []

    k_range = [1, 3, 5, 7, 10]

    for k in k_range:
        pipe1 = Pipeline([('vect', CountVectorizer()), ('tfidf',
                                                        TfidfTransformer()), ('model', KNeighborsClassifier(n_neighbors=k, weights='uniform'))])
        tmp = []
        plotted = False
        for train, test in kf.split(x):
            model_knn = pipe1.fit(x[train], y[train])
            knn_pred = model_knn.predict(x[test])
            tmp.append(mean_squared_error(y[test], knn_pred))

        mean_error.append(np.array(tmp).mean())
        std_error.append(np.array(tmp).std())

    plt.errorbar(k_range, mean_error, yerr=std_error)
    plt.xlabel('k value')
    plt.ylabel('Mean Square Error')
    plt.title("k value for KNN performance")
    plt.show()


def print_confusion_matrix(X_train, X_test, Y_train, Y_test, Z_train, Z_test):
    # Baseline Model
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, Y_train)
    dummy_pred = dummy.predict(X_test)

    baseline = confusion_matrix(Y_test, dummy_pred)
    print("Baseline Model\n", baseline)
    print("Accuarcy:", accuracy_score(Y_test, dummy_pred))

    # Logisitic Regression
    pipe1 = Pipeline([('vect', CountVectorizer()), ('tfidf',
                                                    TfidfTransformer()), ('model', LogisticRegression(solver='lbfgs', C=10))])

    model_lr = pipe1.fit(X_train, Y_train)
    lr_pred = model_lr.predict(X_test)
    lr = confusion_matrix(Y_test, lr_pred)
    print("Logistic Regression\n", lr)
    print("Accuracy:", accuracy_score(Y_test, lr_pred))

    # Knn Model
    pipe2 = Pipeline([('vect', CountVectorizer()), ('tfidf',
                                                    TfidfTransformer()), ('model', KNeighborsClassifier(n_neighbors=3, weights='uniform'))])

    model_knn = pipe2.fit(X_train, Y_train)
    knn_pred = model_knn.predict(X_test)

    knn = confusion_matrix(Y_test, knn_pred)
    print("KNN Classifier\n", knn)
    print("Accuracy:", accuracy_score(Y_test, knn_pred))


def roc_curves(X_train, X_test, Y_train, Y_test, Z_train, Z_test):

    # Baseline Model
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, Y_train)
    base_y_scores = dummy.predict_proba(X_test)
    fpr_baseline, tpr_baseline, _ = roc_curve(Y_test, base_y_scores[:, 1])
    # Logistic Regression
    pipe1 = Pipeline([('vect', CountVectorizer()), ('tfidf',
                                                    TfidfTransformer()), ('model', LogisticRegression(solver='lbfgs', C=10))])

    model_lr = pipe1.fit(X_train, Y_train)
    lr_pred = model_lr.predict(X_test)
    lr = confusion_matrix(Y_test, lr_pred)
    lr_y_scores = model_lr.predict_proba(X_test)
    fpr_lr, tpr_lr, _ = roc_curve(Y_test, lr_y_scores[:, 1])

    plt.plot(fpr_baseline, tpr_baseline, color='green')
    plt.plot(fpr_lr, tpr_lr, color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(["Basline", "Logistic Regression"])
    plt.title('ROC Curves')
    plt.show()
    # Knn Model
    pipe2 = Pipeline([('vect', CountVectorizer()), ('tfidf',
                                                    TfidfTransformer()), ('model', KNeighborsClassifier(n_neighbors=3, weights='uniform'))])

    model_knn = pipe2.fit(X_train, Y_train)
    knn_pred = model_knn.predict(X_test)
    knn_y_scores = model_knn.predict_proba(X_test)
    fpr_knn, tpr_knn, _ = roc_curve(Y_test, knn_y_scores[:, 1])

    plt.plot(fpr_baseline, tpr_baseline, color='green')
    plt.plot(fpr_knn, tpr_knn, color='yellow')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(["Basline", "KNN"])
    plt.title('ROC Curves')
    plt.show()


def main():
    Fake_Detector = FakeNewsDetector()
    df = pd.read_csv("dataset.csv")
    cleaned_dataset = Fake_Detector.clean_data(df)

    dataset = cleaned_dataset[['Article', 'Resource', 'Label']]
    print(dataset.head())
    X = dataset.iloc[:, 0]
    Z = dataset.iloc[:, 1]
    Y = dataset.iloc[:, 2]
    # baseline_model(X, Y, Z)
    # sns.countplot(dataset['Label'])
    # plt.show()
    # data = {'resource': Z,
    #         'label': Y}
    # df = pd.DataFrame(data)
    # sns.countplot(x='label', hue='resource',
    #               data=df, palette='deep')
    # plt.title("Plot of the Dataset")
    # plt.xlabel("Label")
    # plt.ylabel("No. of Articles")
    # plt.show()
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, Y, Z, test_size=0.2, random_state=1)

    # logistic_Regression(X_train, X_test, Y_train, Y_test, Z_train, Z_test)
    # logistic_Regression_crossval(X, Y, Z)

    # knn_classifier(X_train, X_test, Y_train, Y_test, Z_train, Z_test)
    # knn_classifier_crossval(X, Y, Z)

    print_confusion_matrix(X_train, X_test, Y_train, Y_test, Z_train, Z_test)
    roc_curves(X_train, X_test, Y_train, Y_test, Z_train, Z_test)


if __name__ == '__main__':
    main()
