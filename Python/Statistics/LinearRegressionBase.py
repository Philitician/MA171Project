import numpy as np
import matplotlib.pylab as plt

class LinearRegressionBase:
    def __init__(self, location, X, Y, xlabel, ylabel):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.X = X
        self.Y = Y
        self.location = location
        self.n = len(X)
        Sx = sum(X)
        self.mu_x = Sx / self.n
        self.SSx = self.getSSx(X, self.mu_x, self.n)


    def getSSx(self, X, mu_x, n):
        squaredX = np.square(X)
        return sum(squaredX) - (n * (mu_x**2))

    def getBeta(self, XTX, XTy):
        inverse = np.linalg.inv(XTX)
        return inverse @ XTy

    def getError(self, XTy, B):
        yTy = self.Y @ self.Y
        error = yTy - (B @ XTy)
        print("Error: {}".format(error))
        return error

    def calcLinRegSimple(self, alpha, beta):
        return alpha + (beta * self.X)

    def plotLine(self, y):
        plt.grid(True)
        plt.plot(self.X, self.Y, 'k.', label='Scatter plot')
        plt.plot(self.X, y, label='Regression line')
        plt.title(self.location)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        plt.show()