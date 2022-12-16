import numpy as np
import matplotlib.pyplot as plt

#Newt Raph

def func(x):
    f = x**4 - 5*x**3 - 2*x**2 + 24*x
    return f

def dfunc(x):
    f = 4*x**3 - 15*x**2 - 4*x + 24
    return f

def d2func(x):
    f = 12*x**2 - 30*x - 4
    return f


a = 0
b = 1


def newt_raph(x1, tol):
    x0 = 0
    it = 0
    while(abs(x1 - x0) >= tol):
        x0 = x1
        x1 = x0 - dfunc(x0)/d2func(x0)
        it += 1
    
    return x0, func(x0), it


xopt, fopt, it = newt_raph(1, 0.01)

print("xopt = ", xopt)
print("fopt = ", fopt)
print("it = ", it)

x = np.linspace(-1, 5, 100)
fX = func(x)

plt.plot(x, fX, 'red')
plt.scatter(xopt, fopt, c = 'black')
plt.title("Newt-Raph")
plt.show()
