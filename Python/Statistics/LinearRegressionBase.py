import numpy as np
import matplotlib.pylab as plt
from scipy.stats import t

class LinearRegressionBase:
    def __init__(self, X, Y, title, xlabel, ylabel):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.X = X
        self.Y = Y
        self.Xmin = np.min(X)
        self.Xmax = np.max(X)
        self.title = title
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

    def getMu(self, B):
        return B[0] + (B[1] * self.X)

    def getSigma(self, s1, pred_number):
        s = s1 * (np.sqrt( pred_number + (1 / self.n) + ((self.X - self.mu_x) ** 2) * (1 / self.SSx)))
        return s

    def b_pdf(self, m, s, v1):
        s = s * np.sqrt(1 / self.SSx)
        b = []
        x_axis = np.linspace(self.Xmin, self.Xmax)
        for x in x_axis:
            b.append(t.pdf(x, m, s, v1))
        return np.array(b)

    def createLinReg(self, alpha, beta):
        return alpha + (beta * self.X)

    def plotLine(self, y):
        plt.grid(True)
        plt.plot(self.X, self.Y, 'k.', label='Scatter plot')
        plt.plot(self.X, y, label='Regression line')
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        filename = 'Reg_{}_{}'.format(self.xlabel, self.ylabel)
        plt.savefig('Figures/Regression/{}.png'.format(filename))
        plt.show()

    def gamma_pdf(self, k, lam, x):
        upper = (lam * x)**(k - 1)
        lower = np.math.factorial(k - 1)
        return (upper / lower) * k * np.exp(-lam * x)