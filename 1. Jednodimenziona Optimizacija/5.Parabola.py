import numpy as np
import matplotlib.pyplot as plt

def func(x):
    f = -1*(x**4 - 5*x**3 - 2*x**2 + 24*x)
    return f

def parabola(x1, x3, tol):
    #x = [0, 1, 2]^T
    X = np.array([x1, (x1+x3)/2, x3]).transpose()
    pom = np.ones(3).transpose()
    #y = [x^0, x^1, x^2]
    Y = np.array([pom, X, X*X]).transpose()
    F = np.copy(func(X))
    
    #f(x) = a + bx + cx**2
    #linalg.solve => Y*x = F => x = Y / F
    abc = np.linalg.solve(Y, F)
    #f(x) = -28x + 10x**2
    print(abc)
    #xopt = -b/2c
    x = -abc[1]/(2*abc[2])
    #fopt
    fX = func(x)
    n = 0

    #ceo postupak funkcionise sve dok |f(opt) - y(opt)| < tolerancije
    #np.dot radi sledece => sum(NizA * NizB)
    while np.abs(np.dot([1, x, x**2], abc) - fX) > tol:
        if(x > X[1]) and (x < X[2]):
            if(fX < F[1]) and (fX < F[2]):
                X = np.array([X[1], x, X[2]])
            elif(fX > F[1]) and (fX < F[2]):
                X = np.array([X[0], X[1], x])
            else:
                print("GRESKA")
        elif(x > X[0]) and (x < X[1]):
            if(fX < F[0]) and (fX < F[1]):
                X = np.array([X[0], x, X[1]])
            elif(fX > F[1]) and (fX < F[0]):
                X = np.array([x, X[1], X[2]])
            else:
                print("GRESKA")
        else:
            print("X LEZI VAN GRANICA")

        pom = np.ones(3).transpose()
        Y = np.array([pom, X, X*X]).transpose()
        F = np.copy(func(X))
        abc = np.linalg.solve(Y, F)
        x = -abc[1]/(2*abc[2])
        fX = func(x)
        n += 1

    return x, fX, n

xopt, fopt, it = parabola(0, 2, 0.0001)

print("xopt = ", xopt)
print("fopt = ", fopt)
print("it = ", it)

x = np.linspace(-1, 5, 100)
fX = func(x)

plt.plot(x, fX, 'red')
plt.scatter(xopt, fopt, c = 'black')
plt.title("Parabola")
plt.show()

