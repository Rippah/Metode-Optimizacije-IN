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

def adam(gradf, x0, gamma, omega, omega2, epsilon, epsilon2, n):
    x = np.array(x0).reshape(len(x0), 1)
    v = np.zeros(shape = x.shape)
    g = np.zeros(shape = x.shape)
    for i in range(n):
        g = g*omega + (1 - omega)*gradf(x)
        v = omega2*v + (1 - omega2)*np.multiply(gradf(x), gradf(x))
        v_zero = np.abs(v/(1 - omega2))
        g_zero = g/(1 - omega)
        x -= gamma*np.ones(shape = gradf(x).shape)/np.sqrt(v_zero + epsilon)*g_zero
        if np.linalg.norm(gradf(x)) < epsilon2:
            break
    return x

#func(x), [a, b], korak, koeficijent izmene koraka, koeficijent izmene koraka, tolerancija, tolerancija, opseg
rez = adam(lambda x: grad_func(x), [3, 0.1], 0.091, 0.9, 0.99, 1e-6, 1e-6, 1000)
print("rez = ", rez)
print("fopt = ", func(rez[0], rez[1]))
