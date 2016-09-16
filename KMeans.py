import random
from itertools import cycle
from scipy.spatial.distance import euclidean

import numpy as np
import matplotlib.pyplot as plt


def plot_data(axis, title):
    if axis is None:
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_title(title)
    else:
        ax[axis].grid(True)
        ax[axis].set_aspect('equal')
        ax[axis].set_title(title)
    prop_cycle = cycle(plt.rcParams['axes.prop_cycle'])
    for k, (x, y) in enumerate(mu):
        p = next(prop_cycle)
        if axis is None:
            ax.plot(x, y, 'o', ms=8, mew=2.0, zorder=2.1, color=p['color'])
        else:
            ax[axis].plot(x, y, 'o', ms=8, mew=2.0, zorder=2.1, color=p['color'])
        x, y = X[z == k].T
        if axis is None:
            ax.plot(x, y, '.', color=p['color'])
        else:
            ax[axis].plot(x, y, '.', color=p['color'])


def kmeans(X, z, mu, clusters, iterations):
    for iteration in range(iterations):
        for i in range(N):
            distance = np.zeros(K)
            for k in range(clusters):
                distance[k] = euclidean(X[i], mu[k])
            z[i] = np.argmin(distance)
        for k in range(clusters):
            mu[k] = np.average(X[z == k], axis=0)


if __name__ == "__main__":
    K = 3
    X = np.loadtxt('dataset/X.tsv', delimiter='\t')
    N, D = X.shape
    mu = random.sample(X, 3)
    z = np.random.randint(K, size=N)
    f, ax = plt.subplots(1, 2)
    plot_data(0, 'Original Data')
    kmeans(X, z, mu, clusters=K, iterations=10)
    plot_data(1, 'K-means')
    plt.show()
