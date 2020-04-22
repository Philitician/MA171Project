import numpy as np
import stat

class LinearRegression:
    def __init__(self, location, X, Y):
        self.X = X
        self.Y = Y
        self.location = location
        self.n = len(X)
        Sx = sum(X)
        Sy = sum(Y)

        self.mu_x = self.Sx / self.n
