##########
# Linear regression with Ordinary Least Squares (OLS) with various libraries / techniques
# Noah Burget
##########
import numpy as np
import time
class LinearRegressionMethods():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.residuals = []
        self.predictions=[]
        self.weights=np.zeros(X.shape[1]) 
        self.bias=0
    def __str__(self):
        msg = """
              Plain Linear Regression Object. Available methods:
              ols - deterministically solve for best betas with Ordinary least Squares (OLS)
              """
    def _timing(self, start=True, start_time=None):
        if start:
            return time.process_time_ns()
        else:
            return time.process_time_ns() - start_time
    def ols(self, bias=True):
        """
        Solves regression problem via deterministic Ordinary Least Squares (OLS) 

        Args:
            bias (boolean): whether to add a bias column to the matrix (True)
        
        Returns:
            w_hat: learned weights
        """
        if bias==True:
            X_with_bias = np.c_[np.ones((len(self.X), 1)), self.X] # Concatenate objs as columns
        else:
            X_with_bias = self.X
        st = self._timing()
        w_hat = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(self.y)
        # calculate timing
        time_ns = self._timing(start=False, start_time=st)
        time_s = time_ns/1e9
        print("Execution time : {}ns == {}s".format(time_ns, time_s))
        print('Intercept: {} Weights: {}'.format(w_hat[0], w_hat[1:]))
        return w_hat
    def mse_gradient_descent(self, num_iters=500, learning_rate=0.05):
        """
        Finds ideal weights for a matrix (self.X) and target value (self.y) assuming a linear relationship y=w1x1 + w2x2 + ... xnxn

        Args:
            num_iters (int): number of ierations to do gradient descent for (default=500)
            learning_rate (float): step size, how fast we descend towards the minimum (default=0.05)

        Returns:
            tuple of learned weights, bias
        """
        n_feats = self.X.shape[1]
        # Initialize weights as zeroes
        st = self._timing()
        for i in range(0,num_iters):
            # Predict
            yhat = np.dot(self.X, self.weights) + self.bias
            # Keep track of residuals in the object 
            self.residuals = np.abs(yhat - self.y)
            # Calculate gradients with respect to weights and bias
            d_weights = -(2/len(self.X)) * np.dot(self.X.T, (self.y - yhat))
            d_bias = -(2/len(self.X)) * np.sum(self.y - yhat)
            # Update weights and bias
            self.weights -= learning_rate * d_weights
            self.bias -= learning_rate * d_bias
            # Calculate cost
            cost = np.mean((yhat - self.y)**2)
            #print('Cost for iteration {}: {}'.format(i, cost))
                # calculate timing
        time_ns = self._timing(start=False, start_time=st)
        time_s = time_ns/1e9
        print("Execution time : {}ns == {}s".format(time_ns, time_s))
        # Make predictions
        self.predictions = np.dot(self.X, self.weights) + self.bias
        return self