import numpy as np
import matplotlib.pyplot as plt

def func(x):
    f = -1*(x**4 - 5*x**3 - 2*x**2 + 24*x)
    return f

def zlatni_presek(a, b, tol):
    c = (3 - np.sqrt(5))/2
    x1 = a + c*(b-a)
    x2 = a + b - x1
    n = 1

    while (b-a) > tol:
        n += 1
        
        if func(x1) <= func(x2):
            b = x2
            x1 = a + c*(b-a)
            x2 = a + b - x1
        else:
            a = x1
            x1 = a + c*(b-a)
            x2 = a + b - x2

    if func(x1) <= func(x2):
        xopt = x1
        fopt = func(x1)
    else:
        xopt = x2
        fopt = func(x2)

    return xopt, fopt, n

xopt, fopt, it = zlatni_presek(0, 3, 0.0001)

print("xopt = ", xopt)
print("fopt = ", fopt)
print("it = ", it)

x = np.linspace(-1, 5, 100)
fX = func(x)
plt.plot(x, fX, 'red')
plt.scatter(xopt, fopt, c = 'black')
plt.title("Zlatni Presek")
plt.show()
