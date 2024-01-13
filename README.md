## Collection of implementations of various statistical / ML techniques and models

#### Python modules
A collection of python modules exists here for training machine learning mdoels using Numpy. 
* `DataGenerator.py` - A module (object) for generating matrices of synthetic data with one or more features (X) and a response variable (y) with a specified relationship. Currently, these are the supported data generation methods: \
  * `generate_linear(intercept, weights)`: Creates random X data from the standard Uniform distribution ([0,1)) **(addt'l distributions TBI)** and a linearly determined response variable, that is $y = β_0 + β_1x_1 + β_2x_2 + ... + β_nx_n$
  * `generate_polynomial(degrees, coefficients)`: Creates random X data from the standard Uniform distribution ([0,1)) and a linearly determined respinse variable *with added polynomial transformations of the feature(s)*, that is $y = β_0 + β_1x_1 + β_1x_1^2 + β_2x_2 + β_2x_2^2 + ... + β_nx_n^k$
    * NOTE: Yes,  `generate_polynomial([1], [[1,1]]) == generate_linear(intercept=1, weights=[1])`
   
 TBA: Neural network implementations (barebones and pytorch)
    
#### Books referenced:
* Peter Dalgaard - Introductory Statistics with R
