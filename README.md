## Collection of implementations of various statistical / ML techniques and models

#### Python modules
A collection of python modules exists here for training machine learning mdoels using Numpy. 
* `DataGenerator.py` - Module for generating synthetic data with one or more features (X) and a response variable (y) generated via the functions highlighted below. Currently, these are the supported data generation methods:
  * `DataGenerator.x` - The data to be used. Currently, the X data is generated from the Uniform Distribution, by default the Standard Uniform `U[0,1]`. This can be changed by setting the `domain` argument to a tuple like `(min,max)` when instantiating a `DataGenerator` object.
  * `generate_linear(intercept, weights)`: $y = β_0 + β_1x_1 + β_2x_2 + ... + β_nx_n$
  * `generate_polynomial(degrees, coefficients)`: *with added polynomial transformations of the feature(s)*, for example, if `coefficient=2`, $y = β_0 + β_1x_1 + β_1x_1^2 + β_2x_2 + \beta_{2}x_{2}^{2}+...+\beta_{n}x_{n}^{2}$
  * `generate_synthetic_logisticregression(degrees, coefficients, threshold=0.5)` - Generates polynomial X data up to degree `degrees`, with an added transformation via the sigmoid function $\sigma(x)=\frac{1}{1+e^{-x}}$. These probabilities are then thresholded in a binary manner, the threshold of which can be changed via the `threshold` arg (default=0.5).
    
#### Books referenced:
* Peter Dalgaard - Introductory Statistics with R

#### TODO:
Right now the polynomial models are essentially cheating, since we give it the exact model complexity it needs to predict f. Fix: add a parameter for model complexity; it would represent essentially the number of exponents you need to raise the data to (add addtl cols). 
