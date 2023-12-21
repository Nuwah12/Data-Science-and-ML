##########
# Module for evaluating models
# Noah Burget
##########
from . import *
import numpy as np
def coefficientOfDetermination(m):
    if not isinstance(m, LinearRegression.LinearRegressionMethods) or len(m.residuals)==0:
        raise ValueError('Can only evaluate trained Regression models')
    """
    Coefficient of determination (R^2) for a regression model
    """
    # Calculate the residual sum of squares
    ssres = np.sum(m.residuals**2)
    # Calculate total sum of squares
    sstot = np.sum((m.y - np.mean(m.y))**2)
    # Calculate coefficient of determination
    cod = 1 - (ssres/sstot)
    return cod
    