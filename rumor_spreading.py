import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import geom
from scipy.stats import zipf
from scipy.stats import poisson
from collections import Counter


def generate_distrib(N, rvs):
    sample = rvs(N)
    distrib = Counter(sample)
    return np.array(distrib.keys()), np.array(distrib.values()) / float(N)


def sna(mean_degree, N, alpha):
    i_t = 1. - eps
    s_t = eps
    r_t = 0

    while 1:
        w_is = N * mean_degree * i_t * s_t
        w_sr = N * alpha * mean_degree * s_t * (s_t + r_t)
        tau = 1. / (w_is + w_sr)
        reaction = list(np.random.multinomial(1, tau * np.array([w_is, w_sr]))).index(1)
        if reaction:
            value = min(eps, s_t)
            s_t -= value
            r_t += value
        else:
            value = min(eps, i_t)
            i_t -= value
            s_t += value
        if s_t < eps:
            break
    return i_t, s_t, r_t


def sna_plus(degree_array, proba_array, N, alpha):
    size = len(proba_array)
    i_array = proba_array
    s_array = np.zeros(size)
    r_array = np.zeros(size)
    w = np.zeros(2 * size)

    seed = list(np.random.multinomial(1, proba_array)).index(1)
    i_array[seed] -= eps
    s_array[seed] += eps

    while 1:
        s_total = sum(degree_array * s_array) / mean_degree
        sr_total = sum(degree_array * (s_array + r_array)) / mean_degree
        for index, k in enumerate(degree_array):
            w[index] = N * k * i_array[index] * s_total
            w[index + size] = N * alpha * k * s_array[index] * sr_total
        tau = 1. / sum(w)
        # Reaction is the index of the chosen transition from the probability vector of all transitions
        reaction = list(np.random.multinomial(1, tau * w)).index(1)  # Index is like find
        if reaction < size:
            value = min(eps, i_array[reaction])  # prevent negative probabilities when P(k) < eps
            i_array[reaction] -= value
            s_array[reaction] += value
        else:
            value = min(eps, s_array[reaction % size])
            s_array[reaction % size] -= value
            r_array[reaction % size] += value
        total_s = sum(s_array)
        if total_s < eps:
            break
    return i_array, s_array, r_array


def print_moments(proba, degree):
    print("Minimum degree: %s" % min(degree))
    print("<k>=%s" % sum(proba * degree))
    print("<k^2>=%s" % sum(proba * degree ** 2))
    print("<k^2>-<k>^2=%s" % (sum(proba * degree ** 2) - sum(proba * degree) ** 2))
    print("Sigma=%s" % np.sqrt(sum(proba * degree ** 2) - sum(proba * degree) ** 2))

# ------------------------------

N = np.power(10, 5)
print("N=%s" % N)
alpha = 1
print("alpha=%s" % alpha)
beta = 1. + 1. / alpha
eps = 1. / N
mean_degree = 6

expected_r = brentq(lambda x: 1 - x - np.exp(-beta * x), eps, 1.)
print("Expected R is %s \n" % expected_r)

print('Starting SNA computation')
print("<k>=%s" % mean_degree)
i_v, s_v, r_v = sna(mean_degree, N, alpha)
print("I(t) for SNA is %s" % i_v)
print("S(t) for SNA is %s" % s_v)
print("R(t) for SNA is %s \n" % r_v)

print('Starting SNA+ computation for sigma=0')
degree_array = np.array([mean_degree])
proba_array = np.array([1.])
print_moments(proba_array, degree_array)
i_a, s_a, r_a = sna_plus(degree_array, proba_array, N, alpha)
print("I(t) for SNA+ (s=0) is %s" % (sum(i_a)))
print("S(t) for SNA+ (s=0) is %s" % (sum(s_a)))
print("R(t) for SNA+ (s=0) is %s \n" % (sum(r_a)))

print('Starting SNA+ computation for Exponential distribution')
degree_array, proba_array = generate_distrib(N, lambda x: poisson.rvs(mean_degree-1, size=x, loc=1))
print_moments(proba_array, degree_array)
i_a, s_a, r_a = sna_plus(degree_array, proba_array, N, alpha)
print("I(t) for SNA+ (Exp) is %s" % (sum(i_a)))
print("S(t) for SNA+ (Exp) is %s" % (sum(s_a)))
print("R(t) for SNA+ (Exp) is %s \n" % (sum(r_a)))

print('Starting SNA+ computation for Power law distribution')
gamma = 3.
print("Gamma=%s" % gamma)
degree_array, proba_array = generate_distrib(N, lambda x: zipf.rvs(gamma, size=x, loc=3))
print_moments(proba_array, degree_array)
i_a, s_a, r_a = sna_plus(degree_array, proba_array, N, alpha)
print("I(t) for SNA+ (SF) is %s" % (sum(i_a)))
print("S(t) for SNA+ (SF) is %s" % (sum(s_a)))
print("R(t) for SNA+ (SF) is %s \n" % (sum(r_a)))
