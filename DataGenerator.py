##########
# Class for generating random data
# Noah Burget
##########
import numpy as np
class DataGenerator():
    def __init__(self, num_samples=1000, num_features=1, seed=None, noise_std=1):
        np.random.seed(seed)
        self.n_samples=num_samples
        self.seed=seed
        self.noise_std=noise_std
        self.num_features = num_features
        self.noise = np.random.randn(num_samples) * noise_std # Generate noise from the normal distribtion, times some std
        self.X = np.random.rand(num_samples, num_features) # Generate random X values from uniform distribution [0,1]
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
            #print(vals)
            deg = degrees[i]
            coeffs = coefficients[i]
            c = 0 # Counter for column number in out new polynomial feature matrox
            for j in range(deg+1): # Each degree corresponds to one of the expanded polynomial features, 
                #                       in decreasing order - i.e. if the degree for a given feature is 2, 
                #                       3 weights (coefficients) must be multiplied by that feature
                poly_feat = coeffs[j] * (vals**(deg-j))
                #print(poly_feat)
                polynomial_features[:,c] = poly_feat
                c+=1
        #polynomial_features+=self.noise
        y = np.sum(polynomial_features, axis=1)
        y+=self.noise
        return self.X,y
    def generate_exponential(self, base, exponent):
        y = base ** (exponent * self.X) + self.noise
        return self.X,y
    def generate_sinusoidal(self, amplitude, frequency, phase):
         y = amplitude * np.sin(2 * np.pi * frequency * self.X + phase) + self.noise
         return self.X,y


