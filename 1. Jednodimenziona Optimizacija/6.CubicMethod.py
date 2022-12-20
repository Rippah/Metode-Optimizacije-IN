import numpy as np
import matplotlib.pyplot as plt
import math

def func(x):
    f = -1*(x**4 - 5*x**3 - 2*x**2 + 24*x)
    return f

def dfunc(x):
    f = -1*(4*x**3 - 15*x**2 - 4*x + 24)
    return f

def kubna_metoda(x1, x2, tol):
    #Y = [x1^3, x2^3, dx1^3, dx2^3]
    Y = np.array([
                  [1, x1, x1**2, x1**3],
                  [1, x2, x2**2, x2**3],
                  [0, 1, 2*x1, 3*x1**2],
                  [0, 1, 2*x2, 3*x2**2]
                 ])
    #F = [x1, x2, dx1, dx2]
    F = np.array([func(x1), func(x2), dfunc(x1), dfunc(x2)])
    #f(x) = a + bx +cx**2 + dx**3 => x = F / Y
    abc = np.linalg.solve(Y, F)
    
    #Radi jednostavnosti, b, c, d koeficijente sam izmenio u a,b,c za racunanje kvadratnih formula
    #Odnosno x' = a + 2*b*x + 3*c*x**2
    a = abc[1]
    b = abc[2]
    c = abc[3]
    #D = b^2 - 4*a*c (Matrica 4x4 => D = 4*D)
    D = math.sqrt(4*b**2 - 12*a*c)

    #Posto je u Metodi Parabole ova formula bila xopt = -b/2c
    #Ovde ce jer je kubna funkcija formula biti xopt = -2*b/3*2*c
    xa = (-2*b - D)/(6*c)
    xb = (-2*b + D)/(6*c)

    if func(xa) < func(xb):
        x = xa
    else:
        x = xb

    fX = func(x)
    n = 0

    while np.abs(np.dot([1, x, x**2, x**3], abc) - fX) > tol:

        #Posto smo blizi optimalnom resenju, zameniti udaljeniji clan sa xopt
        if func(xa) < func(xb):
            x2 = x
        else:
            x1 = x
        
        Y = np.array([
                      [1, x1, x1**2, x1**3],
                      [1, x2, x2**2, x2**3],
                      [0, 1, 2*x1, 3*x1**2],
                      [0, 1, 2*x2, 3*x2**2]
                     ])
        F = np.array([func(x1), func(x2), dfunc(x1), dfunc(x2)])

        abc = np.linalg.solve(Y, F)
        a = abc[1]
        b = abc[2]
        c = abc[3]

        D = math.sqrt(4*b**2 - 12*a*c)
        xa = (-2*b - D)/(6*c)
        xb = (-2*b + D)/(6*c)

        if func(xa) < func(xb):
            x = xa
        else:
            x = xb
        
        n += 1
        fX = func(x)
    
    return x, fX, n

xopt, fopt, it = kubna_metoda(0, 2, 0.00001)

print("xopt = ", xopt)
print("fopt = ", fopt)
print("it = ", it)

x = np.linspace(-1, 5, 100)
fX = func(x)

plt.plot(x, fX, 'red')
plt.scatter(xopt, fopt, c = 'black')
plt.title("Cubic Method")
plt.show()


def kubna_metode(x1, x2, tol):
    Y = np.array([
        [1, x1, x1**2, x1**3],
        [1, x2, x2**2, x2**3],
        [0, 1, 2*x1, 3*x1**2],
        [0, 1, 2*x2, 3*x2**2]
    ])

    F = np.array([func(x1), func(x2), dfunc(x1), dfunc(x2)])
    abc = np.linalg.solve(Y, F)
    a = abc[1]
    b = abc[2]
    c = abc[3]

    D = math.sqrt(4*b**2 - 12*a*c)
    xa = (-2*b - D)/(6*c)
    xb = (-2*b + D)/(6*c)

    if func(xa) < func(xb):
        x = xa
    else:
        x = xb

    fX = func(x)
    n = 0

    while np.abs(np.dot([1, x, x**2, x**3], abc) - fX) > tol:

        if func(xa) < func(xb):
            x2 = x
        else:
            x1 = x
        Y = np.array([
            [1, x1, x1**2, x1**3],
            [1, x2, x2**2, x2**3],
            [0, 1, 2*x1, 3*x1**2],
            [0, 1, 2*x2, 3*x2**2]
        ])

        F = np.array([func(x1), func(x2), dfunc(x1), dfunc(x2)])

        abc = np.linalg.solve(Y, F)
        a = abc[1]
        b = abc[2]
        c = abc[3]

        D = math.sqrt(4*b**2 - 12*a*c)
        xa = (-2*b - D)/(6*c)
        xb = (-2*b + D)/(6*c)

        if func(xa) < func(xb):
            x = xa
        else:
            x = xb
        n += 1
        fX = func(x)
    
    return x, fX, n

xopt, fopt, it = kubna_metoda(0, 2, 0.00001)

print("xopt = ", xopt)
print("fopt = ", fopt)
print("it = ", it)

x = np.linspace(-1, 5, 100)
fX = func(x)

plt.plot(x, fX, 'red')
plt.scatter(xopt, fopt, c = 'black')
plt.title("Cubic Method")
plt.show()

