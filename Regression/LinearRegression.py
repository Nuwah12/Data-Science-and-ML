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
              Gradient Descent w/ Mean Squared Error (MSE) - minimizes MSE via gradient descent
              X = {}
              y = {}
              """.format(self.X, self.y)
        return msg
    def _timing(self, start=True, start_time=None):
        if start: return time.process_time_ns()
        else: return time.process_time_ns() - start_time
    def ols(self):
        """
        Solves regression problem via deterministic Ordinary Least Squares (OLS) 

        Args:
            bias (boolean): whether to add a bias column to the matrix (True)
        
        Returns:
            w_hat: learned weights
        """
        st = self._timing()
        w_hat = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
        self.weights = w_hat
        # calculate timing
        time_ns = self._timing(start=False, start_time=st)
        time_s = time_ns/1e9
        print("Execution time : {}ns == {}s".format(time_ns, time_s))
        self.predictions = np.dot(self.X, self.weights) + self.bias
        self.residuals = np.abs(self.predictions - self.y)
        return self
    def mse_gradient_descent(self, num_iters=500, learning_rate=0.05, threshold=None):
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
            weight_gradient = (2/len(self.X)) * np.dot(self.X.T, (yhat - self.y))
            bias_gradient = (2/len(self.X)) * np.sum(yhat - self.y)
            # Update weights and bias
            self.weights -= learning_rate * weight_gradient
            self.bias -= learning_rate * bias_gradient
            # Calculate cost (loss, the Mean Squared Error)
            cost = np.mean((yhat - self.y)**2)
            # If convergance threshold was specified:
            if threshold is None:
                continue
            else:
                if cost > threshold:
                    continue
                else:
                    print('Converged after {} iterations, cost = {}'.format(i, cost))
                    break
        # calculate timing
        time_ns = self._timing(start=False, start_time=st)
        time_s = time_ns/1e9
        print("Execution time : {}ns == {}s".format(time_ns, time_s))
        # Make predictions
        self.predictions = np.dot(self.X, self.weights) + self.bias
        self.residuals = np.abs(self.predictions - self.y)
        return self