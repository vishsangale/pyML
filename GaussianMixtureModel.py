"""
    Implementations of Gibbs Samplers.
    0. Chromatic Gibbs Sampler
    1. Collapsed Gibbs Sampler
"""
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt


def sample_instance_collapsed(i):
    """Sample single instance of Collapsed Gibbs Sampler.
    Args:
        i: Input single sample instance.
    Returns: Drawn samples from current instance.
    """
    p = [0] * K
    for k in range(K):
        p[k] = (float(assign[k] + alpha[k])) * (1.0 / ((2 * np.pi * (sigma ** 2)) ** (D / 2))) * \
               np.exp((-1.0 / (2 * sigma ** 2)) * np.sum(np.square(X[i] - mu[k])))
    p /= sum(p)
    return random_state.multinomial(1, p).argmax()


def sample_mean():
    s = np.zeros((K, D))
    for k in range(K):
        s[k] = np.sum(X[z == k], axis=0)
        mu[k] = random_state.multivariate_normal(np.atleast_1d(s[k]) / (assign[k] + (sigma ** 2 / zeta ** 2)),
                                                 (sigma ** 2 / (assign[k] + (sigma ** 2 / zeta ** 2))) * np.eye(D),
                                                 size=1)


def collapsed_gibbs_sampler(iterations=10):
    """Implementation of Collapsed Gibbs Sampler.
    Args:
        iterations: Number of iterations for sampling.
    """
    for iteration in range(iterations):
        for i in range(N):
            k_i = z[i]
            assign[k_i] -= 1
            z[i] = -1
            m = sample_instance_collapsed(i)
            z[i] = m
            assign[z[i]] += 1
        sample_mean()


def sample_instance_chromatic(i):
    """Sample single instance of Chromatic Gibbs Sampler.
    Args:
        i: Input single sample instance.
    Returns: Drawn samples from current instance.
    """
    p = [0] * K
    for k in range(K):
        p[k] = (1.0 / ((2 * np.pi * (sigma ** 2)) ** (D / 2))) * \
               np.exp((-1.0 / (2 * sigma ** 2)) * np.sum(np.square(X[i] - mu[k])))
    p *= random_state.dirichlet(alpha + assign)
    p /= sum(p)
    return random_state.multinomial(1, p).argmax()


def chromatic_gibbs_sampler(iterations=10):
    """Implementation of Chromatic Gibbs Sampler.
    
    Args:
        iterations: Number of itetrations for sampling.
    Returns: None
    """
    for iteration in range(iterations):
        for i in range(N):
            z[i] = sample_instance_chromatic(i)
        sample_mean()


def plot_data(axis, title):
    ax[axis].grid(True)
    ax[axis].set_aspect('equal')
    ax[axis].set_title(title)
    prop_cycle = cycle(plt.rcParams['axes.prop_cycle'])
    for k, (x, y) in enumerate(mu):
        p = next(prop_cycle)
        ax[axis].plot(x, y, 'o', ms=8, mew=2.0, zorder=2.1, color=p['color'])
        x, y = X[z == k].T
        ax[axis].plot(x, y, '.', color=p['color'])


def random_initialize():
    """
    Initialize sampling parameters, inputs, and outputs.
    """
    global sigma, alpha, zeta, K, D, X, N, z, assign, random_state, mu
    sigma = 1.0
    alpha = [1.0, 1.0, 1.0]
    zeta = 2.0
    K = 3
    D = 2
    X = np.loadtxt('X.tsv', delimiter='\t')
    N = X.shape[0]
    z = np.random.randint(K, size=N)
    assign = np.bincount(z)
    random_state = np.random.RandomState(0)
    mu = np.zeros((K, D))


if __name__ == '__main__':
    random_initialize()
    fig, ax = plt.subplots(1, 4)
    plot_data(0, 'Original')
    collapsed_gibbs_sampler(50)
    plot_data(1, 'Collapsed Gibbs Sampler')
    random_initialize()
    plot_data(2, 'Original again')
    chromatic_gibbs_sampler(50)
    plot_data(3, 'Chromatic Gibbs Sampler')
    plt.show()
