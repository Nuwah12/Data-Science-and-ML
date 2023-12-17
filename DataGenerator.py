##########
# Class for generating random data
# Noah Burget
##########
import numpy as np
class DataGenerator():
    def __init__(self, num_samples=1000, seed=None, noise_std=1):
        np.random.seed(seed)
        self.n_samples=num_samples
        self.seed=seed
        self.noise_std=noise_std
        self.noise = np.random.randn(num_samples, 1) * noise_std # Generate noise from the normal distribtion, times some std
        self.X = np.random.rand(num_samples, 1) # Generate random X values from uniform distribution [0,1]
    def __str__(self):
        msg = """
              DataGenerator object::
              {} samples
              seed={}
              noise std={}
              """.format(self.n_samples, self.seed, self.noise_std)
        return msg
    def generate_linear(self, slope, intercept): # Generate corresponding y values with normally distributed noise
        y = slope * self.X + intercept + self.noise
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

