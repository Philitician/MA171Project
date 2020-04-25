import numpy as np
from scipy.stats import t
from scipy.stats import gamma
import matplotlib.pyplot as plt

from Statistics.LinearRegressionBase import LinearRegressionBase


class LinearRegressionCentered(LinearRegressionBase):
    def __init__(self, X, Y, title, xlabel, ylabel):
        mu = sum(X) / len(X)
        X = X - mu
        super().__init__(X, Y, title, xlabel, ylabel)

        self.XTX = self.getXTX(self.SSx, self.n)
        Sy, Sxy = sum(Y), sum(X*Y)
        self.XTy = np.array([Sy, Sxy])
        self.B = self.getBeta(self.XTX, self.XTy)
        SSe = self.getError(self.XTy, self.B)
        self.S1 = SSe
        self.v1 = self.n - 2
        self.sigma = np.sqrt(self.S1 / self.v1)
        y = self.calcLinReg()
        y2 = self.createLinReg(self.B[0], self.B[1])
        self.plotLine(y)
        self.plotLine(y2)

    def calcLinReg(self):
        mu = self.B[0] + (self.B[1] * self.X)
        sigma = self.sigma * (np.sqrt((1 / self.n) + ((1 / self.SSx) * self.X**2)))
        y = t.pdf(self.X, self.v1, mu, sigma)
        print("Linear Regression: \n {}".format(y))
        return y


    def setCenteredX(self):
        self.X = self.X * self.mu_x

    def getXTX(self, SSx, n):
        return np.array([[n, 0], [0, SSx]])