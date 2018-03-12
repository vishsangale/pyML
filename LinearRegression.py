""" Linear Regression"""
import numpy as np


def calculate_cost(X, y, pred):
    """
    Cost function
    """
    #pred = X.dot(theta)
    from sklearn.metrics import mean_squared_error
    #print pred.shape, y.shape
    sqErrors = (pred - y)

    cost = (1.0 / (2 * len(y))) * sqErrors.T.dot(sqErrors)

    return cost


def apply_gradient_descent(number_of_iterations, alpha, X, y, theta):
    cost = [0 for x in range(number_of_iterations)]
    for i in range(number_of_iterations):
        pred = X.dot(theta)

        for t in range(len(theta)):
            error = (pred - y) * X[:, t]
            theta[t] -= alpha * (1.0 / len(y)) * sum(error)
        cost[i] = calculate_cost(X, y, pred)
    #print cost[-5:]
    return theta


if __name__ == "__main__":
    train = np.loadtxt("dataset/house2.csv", delimiter=",", skiprows=1)
    num_features = 9
    y = train[:, num_features]
    X = train[:, :num_features]
    #print type(X)
    theta = [0 for x in range(num_features)]
    theta = apply_gradient_descent(1000, 0.1, X, y, theta)
    test = np.loadtxt("dataset/house2_test.csv", delimiter=",", skiprows=1)
    X_test = test[:, :num_features]
    #print X_test
    y_test = X_test.dot(theta)
    print y_test
