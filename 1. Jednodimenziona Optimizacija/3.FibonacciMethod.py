import numpy as np
import matplotlib.pyplot as plt

def func(x):
    f = -1*(x**4 - 5*x**3 - 2*x**2 + 24*x)
    return f

def fiboBroj(n):
    if n < 0:
        print("Greska")
    elif n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fiboBroj(n-1) + fiboBroj(n-2)

def fibonaci(a, b, tol):
    n = 1
    while((b-a)/tol) > fiboBroj(n):
        n += 1
    
    x1 = a + (fiboBroj(n-2)/fiboBroj(n))*(b-a)
    x2 = a + b - x1

    for i in range(2, n+1):
        if func(x1) <= func(x2):
            b = x2
            x2 = x1
            x1 = a + b - x2
        else:
            a = x1
            x1 = x2
            x2 = a + b - x1
    
    if func(x1) <= func(x2):
        xopt = x1
        fopt = func(x1)
    else:
        xopt = x2
        fopt = func(x2)

    return xopt, fopt, n

xopt, fopt, it = fibonaci(0, 3, 0.0001)

print("xopt = ", xopt)
print("fopt = ", fopt)
print("it = ", it)

x = np.linspace(-1, 5)
fX = func(x)
plt.plot(x, fX, 'red')
plt.scatter(xopt, fopt, c = 'black')
plt.title("Fibonaci")
plt.show()
