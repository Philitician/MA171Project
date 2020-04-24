import numpy as np
from scipy.stats import t
from scipy.stats import gamma

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
        y = self.calcLinReg()
        y2 = self.calcLinRegSimple(self.B[0], self.B[1])
        #self.plotLine(y)
        self.plotLine(y2)


    def calcLinReg(self):
        mu = self.B[0] + (self.B[1] * (self.X - self.mu_x))
        sigma = self.sigma * (np.sqrt((1 / self.n) + ((1 / self.SSx) * (self.X - self.mu_x)**2)))
        y = t.pdf(self.X, self.v1, mu, sigma)
        print("Linear Regression: \n {}".format(y))
        return y

    def createXMatrix(self):
        newX = []
        for x in self.X:
            newX.append([1, x])
        return np.array(newX)

    def a_pdf(self):
        a = t.pdf(self.X, self.B[0], self.sigma * np.sqrt(1/self.n), self.v1)
        print("a* pdf: {}".format(a))

    def getPrecision(self):
        precision_pdf = gamma.pdf(self.X, self.v1/2, self.S1/2)
        print("Precision is: {}".format(precision_pdf))
        return precision_pdf
