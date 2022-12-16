import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def func(x,y):
    f = 0.01*((x-1)**2+2*(y-1)**2)*((x+1)**2+2*(y+1)**2+0.5)*((x+2)**2+2*(y-2)**2+0.7)
    return f

def grad_func(x):
    M = np.eye(2)
    M = [[1,0], [0,10]]
    return np.matmul(M,x)

def func_hessi(x):
    M = [[1,0], [0,10]]
    return M

def newton(gradf, hess, x0, epsilon, n): #hess je funkcija koja pravi matricu hesijana
    x = np.array(x0).reshape(len(x0),1)
    for k in range(n):
        g = gradf(x)
        hess_eval = hess(x) 
        Hinv = np.linalg.inv(hess_eval)
        xp = np.copy(x) #cuva se prethodno resenje zbog uslova za prekidanje algoritma
        x = x - np.dot(Hinv,g)
        if np.linalg.norm(x-xp)<epsilon: #DRUGACIJI USLOV NEGO U SVIM OSTALIM!
            break
    return x

#func(x), [a, b], korak, tolerancija, tolerancija, opseg
rez = newton(lambda x: grad_func(x), lambda x: func_hessi(x), [1, 2], 1e-6, 100)
print("rez = ", rez)
print("fopt = ", func(rez[0], rez[1]))
