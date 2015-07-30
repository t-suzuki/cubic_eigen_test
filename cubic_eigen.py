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

def cubic_eigen_2phase(m):
    a = -np.trace(m)
    b = -0.5*(np.trace(np.dot(m, m)) - a*a)
    c = -np.linalg.det(m)

    # f(x) = x**3 + a x**2 + b x + c
    def f(x): return x**3 + a*x**2 + b*x + c

    # f'(x) = 3x**2 + 2ax + b
    def fprime(x): return 3*x**2 + 2*a*x + b

    # find two local minima/maxima by f'(x) = 0. let the solutions be p < q.
    # note that f(q) <= 0 <= f(p).
    p = (-a - np.sqrt(a*a - 3*b))/3.0
    q = (-a + np.sqrt(a*a - 3*b))/3.0

    # one "good" root of f(x) = 0 can be found at left(right) of p(q) if q(p) is near the multiple root, respectively.
    # q is near the multiple root
    #  ~ f(p) > -f(q)
    #  <=> f((p + q)/2) > 0
    #  <=> f(-a/3) > 0
    # then, first solve f(x) = 0 using Newton's method at left.

    # starting point: the opposite side of multiple root.
    if f(-a/3) > 0:
        start = p + (p - q)
        print 'left', f(p), f(q)
    else:
        start = q - (p - q)
        print 'right', f(p), f(q)

    def newton(x, n=4):
        for i in range(n):
            x = x - f(x)/fprime(x)
        return x

    # climb to the "good" root.
    x_hat = newton(start)

    # let the root be x_hat and factorize f(x).
    # f(x) = (x - x_hat)(x**2 + alpha*x + beta)
    # a = alpha - x_hat, b = beta - alpha*x_hat, c = -beta*x_hat.
    # => alpha = a + x_hat, beta = b + a*x_hat + x_hat**2
    alpha = a + x_hat
    beta  = b + alpha*x_hat

    # solve g(x) = x**2 + alpha*x + beta = 0
    # root r, s (r < s)
    r = (-alpha - np.sqrt(alpha**2 - 4*beta))/2.0
    s = (-alpha + np.sqrt(alpha**2 - 4*beta))/2.0
    t = x_hat

    # t can be either side! r <= s <= t or t <= r <= s
    x_rst = [r, s, t]
    f_rst = map(f, x_rst)
    print f_rst

    # plot
    fig, axs = plt.subplots(1, 1)
    xs = np.linspace(-10, 10, 100)
    axs.plot(xs, f(xs))
    axs.plot([r, s], map(f, [r, s]), 'o', color='yellow')
    axs.plot([t], [f(t)], 'o', color='red')
    axs.axhline(y=0)
    axs.axvline(x=p, color='red')
    axs.axvline(x=q, color='red')
    axs.axvline(x=start, color='green')
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
    e_my2 = cubic_eigen_2phase(A)
    print 'ref', e_ref
    print 'my', e_my
    print 'my2', e_my2
    print 'diff', np.abs(np.array(e_ref) - np.array(e_my))
    print 'diff2', np.abs(np.array(e_ref) - np.array(e_my2))
    plt.show()

