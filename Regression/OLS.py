##########
# Linear regression with Ordinary Least Squares (OLS) with various libraries / techniques
# Noah Burget
##########
import numpy as np
#from sklearn.LinearModel import LinearRegression
class OLS():
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __str__(self):
        msg = """
              Ordinary Least Squares (OLS) Object. Available methods:
              matrix_sol
              sklearn_sol
              """
    def matrix_sol(self, bias=True):
        ### Add bias term to the X matrix
        if bias==True:
            X_with_bias = np.c_[np.ones((len(self.X), 1)), self.X] # Concatenate objs as columns
        else:
            X_with_bias = self.X
        w_hat = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(self.y)
        return w_hat
