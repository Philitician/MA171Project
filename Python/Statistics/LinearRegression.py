import numpy as np
from scipy.stats import t
from scipy.stats import gamma
import matplotlib.pyplot as plt

from Statistics.LinearRegressionBase import LinearRegressionBase


class LinearRegression(LinearRegressionBase):
    def __init__(self, location, X, Y, xlabel, ylabel):
        super().__init__(location, X, Y, xlabel, ylabel)

        self.X_matrix = self.createXMatrix()
        XTX = self.X_matrix.transpose() @ self.X_matrix
        XTy = self.X_matrix.transpose() @ self.Y
        self.B = self.getBeta(XTX, XTy)
        SSe = self.getError(XTy, self.B)
        self.v1 = self.n - 2
        self.S1 = SSe
        self.sigma = np.sqrt(self.S1 / self.v1)
        self.y = self.createLinReg(self.B[0], self.B[1])
        self.plotLine(self.y)

    def createXMatrix(self):
        newX = []
        for x in self.X:
            newX.append([1, x])
        return np.array(newX)

    def a_pdf(self):
        a = t.pdf(self.X, self.B[0], self.sigma * np.sqrt(1/self.n), self.v1)

    def getPrecision(self):
        precision_pdf = gamma.pdf(self.X, self.v1/2, self.S1/2)
        print("Precision is: {}".format(precision_pdf))
        return precision_pdf

    def CI(self, theta):
        m = self.getMu(self.B)
        st = t.ppf(theta, self.v1)
        s = self.getSigma(self.sigma)
        lower = m - (st * s)
        upper = m + (st * s)
        return np.array([lower, upper])

    def plotCredibility(self, color='r', ci = None):
        if ci is None:
            ci = self.CI(0.05)
        plt.grid(True)
        plt.plot(self.X, self.y, color='black')
        plt.fill_between(self.X, ci[0], ci[1], color='r', alpha=.1)
        plt.title(self.location)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        filename = 'CI_{}_{}'.format(self.xlabel, self.ylabel)
        plt.savefig('Figures/Interval/{}.png'.format(filename))
        plt.show()
