##########
# Linear regression with added regularization terms
# Noah Burget
##########
import numpy as np
import time
from .LinearRegression import LinearRegressionMethods
class RegularizedRegressionMethods(LinearRegressionMethods):
    def __init__(self, X, y, L1Lambda=0, L2Lambda=0):
        super().__init__(X, y)
        self.L1 = L1Lambda 
        self.L2 = L2Lambda 
    def __str__(self):
        msg = """
              Regularized Linear Regression Object. Available methods:
              Gradient Descent w/ Mean Squared Error (MSE) - minimizes MSE via gradient descent
              X = {}
              y = {}
              """.format(self.X, self.y)
        return msg
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
            # Calculate regularization 
            l1_reg = self.L1 * np.sum(np.abs(self.weights))
            l2_reg = self.L2 * np.sum(self.weights**2)
            # Calculate gradients with respect to weight vector and L1/L2 regularization
            weight_gradient = (1/len(self.X)) * np.dot(self.X.T, (yhat - self.y))
            l1_gradient = self.L1 * np.sign(self.weights)
            l2_gradient = 2 * self.L2 * self.weights
            # Update weights with added L1 reg. gradient and bias
            self.weights -= learning_rate * weight_gradient + learning_rate * l1_gradient + learning_rate * l2_gradient
            self.bias -= learning_rate * np.mean(yhat - self.y)
            # Calculate cost (loss, the Mean Squared Error, PLUS the L1 and L2 regularization terms)
            cost = np.mean((yhat - self.y)**2) + l1_reg + l2_reg
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
    
