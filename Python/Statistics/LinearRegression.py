import numpy as np
from scipy.stats import t
from scipy.stats import gamma
import matplotlib.pyplot as plt

from Statistics.LinearRegressionBase import LinearRegressionBase


class LinearRegression(LinearRegressionBase):
    def __init__(self, X, Y, title, xlabel, ylabel):
        super().__init__(X, Y, title, xlabel, ylabel)

        self.X_matrix = self.createXMatrix()
        XTX = self.X_matrix.transpose() @ self.X_matrix
        XTy = self.X_matrix.transpose() @ self.Y
        self.B = self.getBeta(XTX, XTy)
        SSe = self.getError(XTy, self.B)
        self.v1 = self.n - 2
        self.S1 = SSe
        self.sigma = np.sqrt(self.S1 / self.v1)
        b = self.b_pdf(self.B[1], self.sigma, self.v1)
        self.y = self.createLinReg(self.B[0], self.B[1])
        self.plotLine(self.y)

    def createXMatrix(self):
        newX = []
        for x in self.X:
            newX.append([1, x])
        return np.array(newX)

    def a_pdf(self):
        a = t.pdf(self.X, self.B[0], self.sigma * np.sqrt(1/self.n), self.v1)


    def Interval(self, theta, pred_number = 0):
        m = self.getMu(self.B)
        st = t.ppf(theta, self.v1)
        s = self.getSigma(self.sigma, pred_number)
        lower = m - (st * s)
        upper = m + (st * s)
        if pred_number == 0:
            self.CI = np.array([lower, upper])
            return self.CI
        else:
            self.PI = np.array([lower, upper])
            return self.PI

    def precisionDistr(self):
        precision = []
        x_axis = np.linspace(self.Xmin, self.Xmax)
        for x in x_axis:
            precision.append(self.gamma_pdf(self.v1 / 2, self.S1 / 2, x))
        print("Precision is: {}".format(precision))
        return precision

    def plotCredibility(self, ci = None, pred = False, both = False):
        if ci is None:
            ci = self.Interval(0.05)

        
        plt.grid(True)

        plt.plot(self.X, self.y, color='black')

        if pred:
            plt.fill_between(self.X, self.PI[0], self.PI[1], facecolor='OliveDrab', interpolate=True, alpha=.7)
        else:
            plt.fill_between(self.X, ci[0], ci[1], facecolor='Crimson', interpolate=True, alpha=.9)

        plt.title(self.location)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        filename = 'CI_{}_{}'.format(self.xlabel, self.ylabel)
        plt.savefig('Figures/Interval/{}.png'.format(filename))
        plt.show()
