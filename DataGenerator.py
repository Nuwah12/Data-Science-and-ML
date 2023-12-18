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
    def generate_linear(self, intercept, weights): # Generate corresponding y values with normally distributed noise
        if isinstance(weights, list):
            weights = np.array(weights).astype(float)
        num_rows = self.X.shape[0]
        ones_column = np.ones((num_rows))
        X = np.insert(self.X, 0, ones_column, axis=1)
        coeffs = np.insert(weights, 0, intercept, axis=0)
        y = np.dot(X, coeffs)
        y = y + self.noise
        return self.X,y
    def generate_exponential(self, base, exponent):
        y = base ** (exponent * self.X) + self.noise
        return self.X,y
    def generate_polynomial(self, coefficients):
        y = np.polyval(coefficients, self.X.squeeze()) + self.noise
        return self.X,y
    def generate_sinusoidal(self, amplitude, frequency, phase):
         y = amplitude * np.sin(2 * np.pi * frequency * self.X + phase) + self.noise
         return self.X,y
def test():
    dg = DataGenerator(num_samples=2500, num_features=3, seed=42)
    l = dg.generate_linear(intercept=8.5,weights=[5,7,10])
    from Regression.LinearRegression import LinearRegressionMethods
    print(l[0])
    print(l[1].shape)
    lr = LinearRegressionMethods(l[0], l[1])
    p = lr.ols()
    print(p)
if __name__ == "__main__":
    test()

