import sys

import numpy as np

from matplotlib import pyplot as plt
from sklearn import datasets
import pandas as pd
from mlxtend.evaluate import plot_decision_regions


class Perceptron:
    def __init__(self, learning_rate, maximum_iterations):
        self.learning_rate = learning_rate
        self.maximum_iterations = maximum_iterations
        self.weight_vector = []
        self.converged = False

    def initialize_weight_vector(self, size):
        self.weight_vector = np.zeros(size)

    def fit(self, X, y):
        number_of_training_samples = X.shape[1]
        self.initialize_weight_vector(number_of_training_samples)
        iterations = 0
        while not self.converged and iterations < self.maximum_iterations:
            self.converged = True
            for i in range(len(X)):
                if y[i] * np.dot(X[i], self.weight_vector) <= 0:
                    delta_weight = y[i] * self.learning_rate * X[i]
                    self.update_weight_vector(delta_weight)
                    self.converged = False
            iterations += 1
        print "Converged in {0} iterations".format(iterations)

    def update_weight_vector(self, delta_weight):
        self.weight_vector += delta_weight

    def predict(self, X):
        np.sign(np.dot(X, self.weight_vector))


def read_data_set(self):
    return np.genfromtxt(self.data_set, delimiter=",", comments="#")


if __name__ == "__main__":
    arg_learning_rate = sys.argv[1]
    arg_maximum_iterations = sys.argv[2]
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values
    perceptron = Perceptron(float(arg_learning_rate), int(arg_maximum_iterations))
    perceptron.fit(X, y)
    print X
    plot_decision_regions(X, y, clf=perceptron)
    plt.title('Perceptron')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.show()
    print X
    #perceptron.plot(X, y)
    # X_test = iris.data[100:]
    # print perceptron.predict(X_test)
