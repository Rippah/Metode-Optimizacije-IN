import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def func(x,y):
    f = 0.01*((x-1)**2+2*(y-1)**2)*((x+1)**2+2*(y+1)**2+0.5)*((x+2)**2+2*(y-2)**2+0.7)
    return f

def grad_func(x):
    diffx = (0.02*x[0] - 0.02)*((x[0] + 1)**2 + 2*(x[1] + 1)**2 + 0.5)*((x[0] + 2)**2 + 2*(x[1] - 2)**2 + 0.7) + (2*x[0] + 2)*(0.01*(x[0] - 1)**2 + 0.02*(x[1] - 1)**2)*((x[0] + 2)**2 + 2*(x[1] - 2)**2 + 0.7) + (2*x[0] + 4)*(0.01*(x[0] - 1)**2 + 0.02*(x[1] - 1)**2)*((x[0] + 1)**2 + 2*(x[1] + 1)**2 + 0.5)
    diffy = (0.04*x[1] - 0.04)*((x[0] + 1)**2 + 2*(x[1] + 1)**2 + 0.5)*((x[0] + 2)**2 + 2*(x[1] - 2)**2 + 0.7) + (4*x[1] - 8)*(0.01*(x[0] - 1)**2 + 0.02*(x[1] - 1)**2)*((x[0] + 1)**2 + 2*(x[1] + 1)**2 + 0.5) + (4*x[1] + 4)*(0.01*(x[0] - 1)**2 + 0.02*(x[1] - 1)**2)*((x[0] + 2)**2 + 2*(x[1] - 2)**2 + 0.7)
    return np.array([diffx, diffy])


# x = sp.Symbol('x')
# y = sp.Symbol('y')

# f = 0.01*((x-1)**2+2*(y-1)**2)*((x+1)**2+2*(y+1)**2+0.5)*((x+2)**2+2*(y-2)**2+0.7)
# difx = f.diff(x)
# dify = f.diff(y)

# dif2x = difx.diff(x)
# dif2y = difx.diff(y)
# dif3x = dify.diff(x)
# dif3y = dify.diff(y)

# print(dif2x)
# print(dif2y)
# print(dif3x)
# print(dif3y)

def grad_func(x):
    M = np.eye(2)
    M = [[2,0], [0,20]]
    return np.matmul(M,x)

def func_hessi(x):
    M = [[2,0], [0,20]]
    return M

def newton(gradf, hess, x0, epsilon, n): #hess je funkcija koja pravi matricu hesijana
    x = np.array(x0).reshape(len(x0),1)
    for i in range(n):
        Hinv = np.linalg.inv(hess(x))
        xp = np.copy(x) #Cuva se prethodno resenje zbog uslova za prekidanje algoritma
        x = x - np.dot(Hinv, gradf(x))
        if np.linalg.norm(x - xp) < epsilon: #DRUGACIJI USLOV NEGO U SVIM OSTALIM!
            break
    return x
        

#func(x), [a, b], korak, tolerancija, tolerancija, opseg
rez = newton(lambda x: grad_func(x), lambda x: func_hessi(x), [1, 2], 1e-6, 100)
print("rez = ", rez)
print("fopt = ", func(rez[0], rez[1]))
