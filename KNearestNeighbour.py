import numpy as np


class KNearestNeighbour:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, distance_type, K=1):
        num_testing_samples = X.shape[0]
        y_predict = np.zeros(num_testing_samples)
        distance = self.calculate_distance(X, distance_type)
        closest_y = self.y_train[np.argsort(distance)][:, :K]
        for _ in range(num_testing_samples):
            counts = np.bincount(closest_y[_])
            y_predict[_] = np.argmax(counts)
        return y_predict

    def calculate_l2_distance(self, X):
        return np.sqrt(np.sum(np.square(X[:, np.newaxis] - self.X_train), axis=2))

    def calculate_distance(self, X, distance_type='l2'):
        if distance_type == 'l2':
            return self.calculate_l2_distance(X)
        else:
            raise ValueError('Invalid value %s for type of distance' % distance_type)


if __name__ == '__main__':
    X_train = np.array([[1, 2, 3], [4, 5, 6], [3, 2, 1], [4, 6, 5], [2, 3, 1]])
    y_train = np.array([0, 1, 0, 1, 0])
    X_test = np.array([[5, 4, 6], [5, 6, 4], [2, 1, 3]])
    knn = KNearestNeighbour()
    knn.train(X_train, y_train)
    print knn.predict(X_test, distance_type='l2', K=2)
