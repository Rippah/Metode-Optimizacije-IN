import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def func(x,y):
    f = 0.01*((x-1)**2+2*(y-1)**2)*((x+1)**2+2*(y+1)**2+0.5)*((x+2)**2+2*(y-2)**2+0.7)
    return f

x = sp.Symbol('x')
y = sp.Symbol('y')

# f = 0.01*((x-1)**2+2*(y-1)**2)*((x+1)**2+2*(y+1)**2+0.5)*((x+2)**2+2*(y-2)**2+0.7)
# difx = f.diff(x)
# dify = f.diff(y)

# print(difx)
# print(dify)


def grad_func(x):
    diffx = (0.02*x[0] - 0.02)*((x[0] + 1)**2 + 2*(x[1] + 1)**2 + 0.5)*((x[0] + 2)**2 + 2*(x[1] - 2)**2 + 0.7) + (2*x[0] + 2)*(0.01*(x[0] - 1)**2 + 0.02*(x[1] - 1)**2)*((x[0] + 2)**2 + 2*(x[1] - 2)**2 + 0.7) + (2*x[0] + 4)*(0.01*(x[0] - 1)**2 + 0.02*(x[1] - 1)**2)*((x[0] + 1)**2 + 2*(x[1] + 1)**2 + 0.5)
    diffy = (0.04*x[1] - 0.04)*((x[0] + 1)**2 + 2*(x[1] + 1)**2 + 0.5)*((x[0] + 2)**2 + 2*(x[1] - 2)**2 + 0.7) + (4*x[1] - 8)*(0.01*(x[0] - 1)**2 + 0.02*(x[1] - 1)**2)*((x[0] + 1)**2 + 2*(x[1] + 1)**2 + 0.5) + (4*x[1] + 4)*(0.01*(x[0] - 1)**2 + 0.02*(x[1] - 1)**2)*((x[0] + 2)**2 + 2*(x[1] - 2)**2 + 0.7)
    return np.array([diffx, diffy])

def steepest_descent(gradf, x0, gamma, epsilon, n):
    x = np.array(x0).reshape(len(x0), 1)
    for i in range(n):
        x = x - gamma*gradf(x)
        if np.linalg.norm(gradf(x)) < epsilon:
            break
    return x

#func(x), [a, b], korak, tolerancija, opseg
rez = steepest_descent(lambda x: grad_func(x), [1, 2], 0.15, 1e-4, 1000)
print("rez = ", rez)
print("fopt = ", func(rez[0], rez[1]))

