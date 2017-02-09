import numpy as np
import matplotlib.pylab as plt


def conv_1d(f, k):
    f = np.pad(f, (len(k)-1, len(k)-1), 'constant', constant_values=(0, 0))
    conv = np.zeros(len(f) - len(k) + 1)
    for i in range(len(f) - len(k) + 1):
        for j in range(len(k)):
            conv[i] += f[i + j] * k[j]
            print(f[i + j], k[j])
    return conv


f = np.array([3, 1, 8, 6, 3, 9, 5, 1])
k = np.array([0.5, 0.5])

# a) Discrete convolution of signal f with kernel k: s1 = f * k
s1 = conv_1d(f, k)
print('s1:', s1)
""" s1: [ 1.5  2.   4.5  7.   4.5  6.   7.   3.   0.5] """

# b) Compute the discrete convolution of s1 and the kernel k: s2 = s1 ∗ k.
s2 = conv_1d(s1, k)
print('s2:', s2)
""" [ 0.75  1.75  3.25  5.75  5.75  5.25  6.5   5.    1.75  0.25] """

# c) Provide a plot of f, s1 and s2. Briefly describe the effects of
# convolving f with the kernel k. What would be the outcome of convolving
# f with k for n times when n → ∞?
plt.plot(f, label='initial signal')
plt.plot(s1, label='after 1-st convolution')
plt.plot(s2, label='after 2-nd convolution')
plt.legend()

# Simulation with big value of n: we receive gaussian distribution
n, s_inf = 1000, f
for iter in range(n):
    s_inf = conv_1d(s_inf, k)
print("After", n, "convolutions:", s_inf)
plt.plot(s_inf)

"""
This particular kernel smoothes the initial signal, because it does averaging
of 2 neighboring elements. Because of "border effects", when convolution averages
border element with 0, elements from the middle of signal to borders will smoothly
change from some middle peak value to zero on its borders. Thus with big enough n,
the shape of the signal will look like a gaussian distribution.
"""

# d)
k_new = conv_1d(k, k)
print("New kernel:", k_new)
""" New kernel: [ 0.25  0.5   0.25] """

s3 = conv_1d(f, k_new)
print("s3:", s3)
"""
s3 = s2 = [ 0.75,  1.75,  3.25,  5.75,  5.75,  5.25,  6.5 ,  5.  ,  1.75,  0.25]

This concept is called "Associative property".
"""
