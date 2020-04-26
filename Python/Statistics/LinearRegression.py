import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import seaborn as sns


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
        self.y = self.createLinReg(self.B[0], self.B[1])
        self.plotLine(self.y)

    def plotFrequency(self):
        plt.hist(self.X, bins=20)
        plt.title("Confirmation Rate Frequency diagram")
        plt.xlabel("Confirmation Rate")
        plt.ylabel("Frequency")
        plt.savefig('Figures/Frequency/ConfirmationRate.png')
        plt.show()
        print("X frequency share: {}".format(sum(i <= 0.1 for i in self.X) / self.n))

        plt.hist(self.Y, bins=20)
        plt.title("Death Rate Frequency diagram")
        plt.xlabel("Death Rate")
        plt.ylabel("Frequency")
        plt.savefig('Figures/Frequency/DeathRate.png')
        plt.show()
        print("Y frequency share: {}".format(sum(i <= 200 for i in self.Y) / self.n))

    def createXMatrix(self):
        newX = []
        for x in self.X:
            newX.append([1, x])
        return np.array(newX)

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

    def plotCredibility(self, ci = None, pred = False):
        if ci is None:
            ci = self.Interval(0.05)

        plt.grid(True)
        plt.plot(self.X, self.y, label='Linear Regression')

        if pred:
            plt.fill_between(self.X, self.PI[0], self.PI[1], color='g', alpha=.2, label='95% Prediction Interval')
            plt.plot(self.X, self.Y, 'k.', label='Scatter plot')

        plt.fill_between(self.X, self.CI[0], self.CI[1], color='r', alpha=.4, label='95% Credible Interval')

        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        filename = 'CI'
        if pred:
            filename = 'PI'
        filename += '_{}_{}'.format(self.xlabel, self.ylabel)
        plt.savefig('Figures/Interval/{}.png'.format(filename))
        plt.show()
