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

def adagrad_gradient(gradf, x0, gamma, omega, epsilon, epsilon2, n):
    x = [np.array(x0).reshape(len(x0), 1)]
    v = np.zeros(shape = x[-1].shape)
    g = [np.zeros(shape = x[-1].shape)]
    for i in range(n):
        #Castuje u array
        G = np.asarray(gradf(x[-1]))
        g.append(omega*g[-1] + (1-omega)*np.multiply(G, G))
        v = gamma*np.ones(shape = G.shape) / np.sqrt(g[-1] + epsilon)*G
        x.append(x[-1] - v)
        if np.linalg.norm(gradf(x)) < epsilon2:
            break
    return x, g

#func(x), [a, b], korak, koeficijent izmene koraka, tolerancija, tolerancija, opseg
rez, g = adagrad_gradient(lambda x: grad_func(x), [3, 0.1], 0.1, 0.9, 1e-6, 1e-6, 100)
print("rez = ", rez)
print("g = ", g)
print("fopt = ", func(rez[0], rez[1]))
