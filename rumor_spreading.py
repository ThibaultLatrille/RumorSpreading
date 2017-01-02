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
    s_list, i_list, r_list, t_list = [], [], [], []
    t = 0.
    t_tmp = 0.

    i_t = 1. - eps
    s_t = eps
    r_t = 0.
    while 1:
        w_is = N * mean_degree * i_t * s_t
        w_sr = N * alpha * mean_degree * s_t * (s_t + r_t)
        tau = 1. / (w_is + w_sr)
        t += tau
        t_tmp += tau
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
        if t_tmp > 0.05:
            t_tmp -= 0.05
            i_list.append(i_t)
            s_list.append(s_t)
            r_list.append(r_t)
            t_list.append(t)
    i_list.append(i_t)
    s_list.append(s_t)
    r_list.append(r_t)
    t_list.append(t)
    return t_list, i_list, s_list, r_list


def sna_plus(degree_array, proba_array, N, alpha, init=None):
    print("Starting SNA with alpha: %s" % alpha)
    s_list, i_list, r_list, t_list = [], [], [], []
    t = 0.
    t_tmp = 0.

    size = len(proba_array)
    i_array = proba_array
    s_array = np.zeros(size)
    r_array = np.zeros(size)
    w = np.zeros(2 * size)

    if init == None:
        seed = list(np.random.multinomial(1, proba_array)).index(1)
    else:
        seed = init
    i_array[seed] -= eps
    s_array[seed] += eps

    while 1:
        s_total = sum(degree_array * s_array) / mean_degree
        sr_total = sum(degree_array * (s_array + r_array)) / mean_degree
        for index, k in enumerate(degree_array):
            w[index] = N * k * i_array[index] * s_total
            w[index + size] = N * alpha * k * s_array[index] * sr_total
        tau = 1. / sum(w)
        t += tau
        t_tmp += tau
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
        if t_tmp > 0.05:
            t_tmp -= 0.05
            i_list.append(sum(i_array))
            s_list.append(sum(s_array))
            r_list.append(sum(r_array))
            t_list.append(t)
    i_list.append(sum(i_array))
    s_list.append(sum(s_array))
    r_list.append(sum(r_array))
    t_list.append(t)
    return t_list, i_list, s_list, r_list, r_array


def print_moments(proba, degree):
    print("Minimum degree: %s" % min(degree))
    print("<k>=%s" % sum(proba * degree))
    print("<k^2>=%s" % sum(proba * degree ** 2))
    print("<k^2>-<k>^2=%s" % (sum(proba * degree ** 2) - sum(proba * degree) ** 2))
    print("Sigma=%s" % np.sqrt(sum(proba * degree ** 2) - sum(proba * degree) ** 2))


def plot_isr(t_v, i_v, s_v, r_v, text="My plot"):
    plt.subplots()
    plt.plot(t_v, i_v, 'b', linewidth=2, label=r'$I(t)$')
    plt.plot(t_v, s_v, 'g', linewidth=2, label=r'$S(t)$')
    plt.plot(t_v, r_v, 'r', linewidth=2, label=r'$R(t)$')
    plt.xlabel("t")
    plt.title(text)
    plt.legend()


def plot_ignorant(N):
    plt.subplots()
    eps = 1./ N
    alpha = 1.5
    degree_array, proba_array = generate_distrib(N, lambda x: poisson.rvs(5., size=x, loc=1))
    i_arrays = [sna_plus(degree_array.copy(), proba_array.copy(), N, alpha)[4] for _ in range(10)]
    i_array = np.mean(i_arrays, axis=0)
    m = np.ma.masked_where(i_array < eps, i_array)
    d_m = np.ma.masked_where(np.ma.getmask(m), degree_array)
    i_m = np.ma.masked_where(np.ma.getmask(m), i_array)
    p_m = np.ma.masked_where(np.ma.getmask(m), proba_array)
    plt.scatter(d_m, p_m, marker='o', s=40, linewidths=1, c='b', label=r'$i_k (t=0 )$')
    plt.scatter(d_m, i_m, marker='^', s=40, linewidths=1, c='r', label=r'$r_k (t=\infty )$')
    plt.ylim((eps/2, max(max(i_m), max(p_m))*2))
    plt.xlim(0, max(degree_array) + 1 )
    plt.yscale("log")
    plt.xlabel(r'$k$')
    plt.ylabel(r'$P(k)$')
    plt.legend()
    plt.title("Density of ignorants at the beginning (blue) \n and stiflers at the end (red)")


def plot_delay(N):
    plt.subplots()
    degree_array, proba_array = generate_distrib(N, lambda x: zipf.rvs(3., size=x, loc=1))
    t_v, i_v, s_v, r_v, _ = sna_plus(degree_array.copy(), proba_array.copy(), N, alpha, init=0)
    plt.plot(t_v, r_v, 'orange', linewidth=2, label=r'$k_i= %s$' % degree_array[0])

    t_v, i_v, s_v, r_v, _ = sna_plus(degree_array.copy(), proba_array.copy(), N, alpha, init=4)
    plt.plot(t_v, r_v, 'red', linewidth=2, label=r'$k_i= %s$' % degree_array[4])

    t_v, i_v, s_v, r_v, _ = sna_plus(degree_array.copy(), proba_array.copy(), N, alpha, init=-1)
    plt.plot(t_v, r_v, 'black', linewidth=2, label=r'$k_i= %s$' % degree_array[-1])

    plt.xlabel("t")
    plt.ylabel(r'$R(t)$')
    plt.title("Stiflers as a function of time for different degree \n of the initial spreader ($k_i$)")
    plt.legend()


def plot_r_inf(N):
    eps = 1. / N
    plt.subplots()
    alpha_list = np.linspace(0.1, 3, 20)
    r_list = [brentq(lambda x: 1 - x - np.exp(-(1. + 1. / alpha) * x), eps, 1.) for alpha in alpha_list]
    r_mf_list = [sna(mean_degree, N, alpha)[3][-1] for alpha in alpha_list]

    degree_array, proba_array = generate_distrib(N, lambda x: zipf.rvs(3., size=x, loc=1))
    r_epx_list = [sna_plus(degree_array.copy(), proba_array.copy(), N, alpha)[3][-1] for alpha in alpha_list]

    degree_array, proba_array = generate_distrib(N, lambda x: zipf.rvs(2., size=x, loc=1))
    r_pl_list = [sna_plus(degree_array.copy(), proba_array.copy(), N, alpha)[3][-1] for alpha in alpha_list]

    plt.plot(alpha_list, r_list, 'r', linewidth=2, label=r'Analytical')
    plt.plot(alpha_list, r_mf_list, 'rs', ms=9., label=r'Mean field')
    plt.plot(alpha_list, r_epx_list, 'r^', ms=9., label=r'$P(k) \sim k^{-3}$')
    plt.plot(alpha_list, r_pl_list, 'ro', ms=9., label=r'$P(k) \sim k^{-2}$')

    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$r(t={\infty})$')
    plt.legend()
    plt.title("Number of stiflers at the end of the process \n as a function of " + r'$\alpha$')

# ------------------------------

N = np.power(10, 4)
print("N=%s" % N)
alpha = 1
print("alpha=%s" % alpha)
eps = 1. / N

expected_r = brentq(lambda x: 1 - x - np.exp(-(1. + 1. / alpha) * x), eps, 1.)
print("Expected R is %s \n" % expected_r)

mean_degree = 6
t_v, i_v, s_v, r_v = sna(mean_degree, N, alpha)
plot_isr(t_v, i_v, s_v, r_v, r'Rumor spreading process as a function of $t$')

plot_r_inf(N)

plot_delay(N)

plot_ignorant(N)

plt.show()
