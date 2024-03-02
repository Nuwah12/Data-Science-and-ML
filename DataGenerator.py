##########
# Class for generating random data
# Noah Burget
##########
import numpy as np
import math
class DataGenerator():
    def __init__(self, num_samples=1000, num_features=1, domain=(0,1), seed=None, noise_std=1):
        """
        Initialize a DataGenerator object
        Arguments:
            num_samples - no. samples to have in dataset
            num_features - no. features to have in dataset
            domain - range to draw Uniform random sample from (default: [0,1))     
            seed - random seed (default None)
            noise_std - std. devation of guassian noise (default: 1) 
        The samples drawn are (by default) from the standard uniform deviation [0,1]
        """
        np.random.seed(seed)
        self.n_samples=num_samples
        self.seed=seed
        self.noise_std=noise_std
        self.num_features = num_features
        self.noise = np.random.randn(num_samples) * noise_std # Generate noise from the normal distribtion, times some std
        self.X = (domain[1]-domain[0]) * np.random.random_sample(size=(num_samples,num_features)) + domain[0] # Generate random X values from uniform distribution [0,1]
    def __str__(self):
        msg = """
              DataGenerator object::
              {} features, {} samples
              seed={}
              noise std={}
              """.format(self.num_features, self.n_samples, self.seed, self.noise_std)
        return msg
    def _insert_bias(self):
        num_rows = self.X.shape[0]
        ones_column = np.ones((num_rows))
        X = np.insert(self.X, 0, ones_column, axis=1)
        return X
    def generate_linear(self, intercept, weights): # Generate corresponding y values with normally distributed noise
        if isinstance(weights, list):
            weights = np.array(weights).astype(float)
        X = self._insert_bias() # Insert a bias column into our matrix 
        coeffs = np.insert(weights, 0, intercept, axis=0) # Add the intercept as a weight (B0)
        y = np.dot(X, coeffs)
        y = y + self.noise
        return self.X,y
    def generate_polynomial(self, degrees, coefficients):
        # Determine the number of features we will have after getting polynomial features
        num_poly_features = sum([degree+1 for degree in degrees]) 
        polynomial_features = np.empty((self.n_samples, num_poly_features)) # init new matrix of the right size for our orignial + polynomial feautres
        for i in range(self.num_features): # Loop thru features (columns)
            vals = self.X[:,i]
            deg = degrees[i]
            coeffs = coefficients[i]
            c = 0 # Counter for column number in out new polynomial feature matrox
            for j in range(deg+1): # Each degree corresponds to one of the expanded polynomial features, 
                #                       in decreasing order - i.e. if the degree for a given feature is 2, 
                #                       3 weights (coefficients) must be multiplied by that feature
                polynomial_features[:,c] = vals**(deg-j)
                c+=1
        #polynomial_features+=self.noise
        coefficients = np.array(coefficients).flatten()
        y = np.dot(polynomial_features, coefficients)
        y += self.noise
        return polynomial_features,y
    def generate_synthetic_logisticregression(self, degrees, coefficients, threshold=0.5):
        """
        Function for generating synthetic (response) data for training a logistic regression model
        All we are doing here is taking the response variable from one of the other data generation methods and passing it through a inverse-logit function (standard logistic function)
        sigma(x) = 1/(1+e^{-x})

        parameters:
            degrees - the polynomial degree to apply to each feature. its length must be equal to the numbver of features in your dataset
            coefficients - the coefficients (i.e. true parameters) to apply to each polynomial of each feature. its length must be equal to (degree of polynomial+1) * number of features
            threshold - decision threshold to apply to the sigmoid output (default=0.5)
        """
        x,y = self.generate_polynomial(degrees=degrees, coefficients=coefficients)
        y_prob = self._sigmoid(y)
        self.probabilities = y_prob
        y = [1 if i >= threshold else 0 for i in y_prob]
        return x,y
    def _sigmoid(self, x):
        return 1 / (1 + math.e**x)

