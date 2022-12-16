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

    F = np.linspace(0, 0, len(X))

    for i in range(0, len(X)):
        F[i] = func(X[i])
    
    abc = np.linalg.solve(Y, F)
    print(abc)
    x = -abc[1]/(2*abc[2])
    fx = func(x)
    n = 0

    while np.abs(np.dot([1, x, x**2], abc) - fx) > tol:
        if(x > X[1]) and (x < X[2]):
            if(fx < F[1]) and (fx < F[2]):
                X = np.array([X[1], x, X[2]])
                F = np.array([F[1], fx, F[2]])
            elif(fx > F[1]) and (fx < F[2]):
                X = np.array([X[0], X[1], x])
                F = np.array([F[0], F[1], fx])
            else:
                print("GRESKA")
        elif(x > X[0]) and (x < X[2]):
            if(fx < F[0]) and (fx < F[1]):
                X = np.array([X[0], x, X[2]])
                F = np.array([F[0], fx, F[2]])
            elif(fx > F[1]) and (fx < F[0]):
                X = np.array([x, X[1], X[2]])
                F = np.array([fx, F[1], F[2]])
            else:
                print("GRESKA")
        else:
            print("X LEZI VAN GRANICA")

        pom = np.ones(3).transpose()
        Y = np.array([pom, X, X*X]).transpose()
        F = np.linspace(0, 0, len(X))

        for i in range(0, len(X)):
            F[i] = func(X[i])
        abc = np.linalg.solve(Y, F)
        x = -abc[1]/(2*abc[2])
        fx = func(x)
        n += 1

    return x, fx, n

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

