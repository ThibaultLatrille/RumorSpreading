import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
mean_degree = 6
print("<k>=%s" % mean_degree)
N = np.power(10, 4)
print("N=%s" % N)
alpha = 0.5
print("alpha=%s" % alpha)

beta = 1. + 1. / alpha
eps = 1. / N
expected_r = brentq(lambda x: 1 - x - np.exp(-beta * x), eps, 1.)
print("Expected R is %s" % expected_r)

i_t = 1. - eps
s_t = eps
r_t = 0
t = 0

while 1:
    w_is = N * mean_degree * i_t * s_t
    w_sr = N * alpha * mean_degree * s_t * (s_t + r_t)
    tau = 1. / (w_is + w_sr)
    reaction = list(np.random.multinomial(1, tau * np.array([w_is, w_sr]))).index(1)
    if reaction == 0:
        i_t -= eps
        s_t += eps
    else:
        s_t -= eps
        r_t += eps
    if s_t < eps:
        break

print("R(t) at the end of the computation is %s" % r_t)


