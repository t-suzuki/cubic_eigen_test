#!env python
# https://en.wikipedia.org/wiki/Eigenvalue_algorithm
# https://en.wikipedia.org/wiki/Cubic_function#Roots_of_a_cubic_function
# https://en.wikipedia.org/wiki/Eigenvalue_algorithm

import matplotlib.pyplot as plt
import numpy as np

def cubic_eigen(m):
    # det(x I - m) = 0
    # <=> x**3 - x**2 * tr(m) - x/2*(tr(m**2) - tr(m)**2) - det(m) = 0
    # <=> x**3 + a x**2 + b x + c = 0
    a = -np.trace(m)
    b = -0.5*(np.trace(np.dot(m, m)) - a*a)
    c = -np.linalg.det(m)

    # f(x) = x**3 + a x**2 + b x + c
    def f(x): return x**3 + a*x**2 + b*x + c

    # f'(x) = 3x**2 + 2ax + b
    def fprime(x): return 3*x**2 + 2*a*x + b

    # f'(x) = 0
    p = (-a + np.sqrt(a*a - 3*b))/3.0
    q = (-a - np.sqrt(a*a - 3*b))/3.0

    # starting point for Newton's method
    r, s, t = p + (p - q), (p + q)/2, q - (p - q)

    def newton(x, n=4):
        for i in range(n):
            x = x - f(x)/fprime(x)
        return x

    # climb to the root.
    x_rst = map(newton, (r, s, t))
    f_rst = map(f, x_rst)
    print f_rst

    # plot
    fig, axs = plt.subplots(1, 1)
    xs = np.linspace(-10, 10, 100)
    axs.plot(xs, f(xs))
    axs.plot(x_rst, f_rst, 'o', color='yellow')
    axs.axhline(y=0)
    axs.axvline(x=p, color='red')
    axs.axvline(x=q, color='red')
    axs.axvline(x=r, color='green')
    axs.axvline(x=s, color='green')
    axs.axvline(x=t, color='green')
    axs.set_ylim(sorted(map(lambda x: f(x)*2, [p, q])))

    return sorted(x_rst)

def cubic_eigen_ref(m):
    a, b = np.linalg.eigh(m)
    return sorted(a)

if __name__=='__main__':
    A = np.random.randn(3, 3)
    A = A + A.T # symmetrize
    print A
    e_ref = cubic_eigen_ref(A)
    e_my = cubic_eigen(A)
    print 'ref', e_ref
    print 'my', e_my
    print 'diff', np.abs(np.array(e_ref) - np.array(e_my))
    plt.show()

