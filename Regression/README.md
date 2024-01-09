### `Regression` package contains modules for training regression models on synthetic or real data.
  * `LinearRegression.py` multiple classes for training plain (no regularization terms added) linear regression models. Currently, this class supports training by Ordinary Least Squares (OLS) and Gradient Descent with Mean Squared Error as the cost function for linear polynomial data of any degree (Support for linearization of exponential functions TBI)
  * `RegularizedRegression.py` classes for training linear regression models with added L1 (Lasso), L2 (Ridge), or L1+L2 (Elastic Net) regularization penalty terms.
    * NOTE: Yes,  `RegularizedRegressionMethods.mse_gradient_descent(l1_lambda=0, l2_lambda=0) == LinearRegressionMethods.mse_gradient_descent()`
    * L1 or L2 (or some combination of the two) penalty terms, multiplied by their respecitve weight (`lambda`) parameters are added to the cost function of each iteration of training  
