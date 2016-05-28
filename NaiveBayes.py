#!/usr/bin/python
from sklearn import datasets
from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    def __init__(self, train_features, train_labels, test_features):
        self.test_features = test_features
        self.predicted_labels = None
        self.train_features = train_features
        self.train_labels = train_labels
        self.clf = None

    def get_accuracy(self, test_labels):
        return accuracy_score(self.predicted_labels, test_labels)

    def train(self):
        self.clf = GaussianNB()
        self.clf.fit(self.train_features, self.train_labels)

    def predict(self):
        self.predicted_labels = self.clf.predict(self.test_features)


if __name__ == "__main__":
    iris = datasets.load_iris()
    features, labels = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
    nb = NaiveBayes(X_train, y_train, X_test)
    nb.train()
    nb.predict()
    print nb.get_accuracy(y_test)
