import numpy as np
import matplotlib.pyplot as plt

def func(x):
    f = x**4 - 5*x**3 - 2*x**2 + 24*x
    return f

def dfunc(x):
    f = 4*x**3 - 15*x**2 - 4*x + 24
    return f

def d2func(x):
    f = 12*x**2 - 30*x - 4
    return f

def secantMethod(x0, x1, tol):
    it = 0
    x2 = x1
    while True:
        if(dfunc(x0) == dfunc(x1)):
            print("Nula!")
            break
        if(abs(x1 - x0) < tol):
            return x2, func(x2), it
        it = it + 1
        x2 = x1 - dfunc(x1)*(x1 - x0)/(dfunc(x1) - dfunc(x0))
        x0 = x1
        x1 = x2

xopt, fopt, it = secantMethod(0, 1, 0.01)

print("xopt = ", xopt)
print("fopt = ", fopt)
print("it = ", it)

x = np.linspace(-1, 5, 100)
fX = func(x)

plt.plot(x, fX, 'red')
plt.scatter(xopt, fopt, c = 'black')
plt.title("Secant Method")
plt.show()
