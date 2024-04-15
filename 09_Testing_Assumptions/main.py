import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean
import numpy as np
import random


style.use("fivethirtyeight")


# positive correlation = positive step #
# negative correlation = negative step #
def create_dataset(how_many, variance, step=2, correlation=False):
    """
    Creating a randomized dataset of any size.

    3 types of datasets can be created:
        1. A dataset with a -positive- correlation.
        2. A dataset with a -negative- correlation.
        3. A dataset with a -no- correlation.
    """
    
    val = 1
    ys = []
    for i in range(how_many):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == "positive":
            val += step
        elif correlation and correlation == "negative":
            val -= step
    
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    """
    Calculate the slope (m) and intercept (b) of the best fit line for a given set of data points.
    
    This function uses the method of least squares to find the best fitting line for the given 
    x (independent variable) and y (dependent variable) data points. The best fit line is described
    by the equation y = mx + b, where 'm' is the slope of the line and 'b' is the y-intercept, 
    indicating where the line crosses the y-axis.
    """

    m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) /
            ((mean(xs)**2) - mean(xs**2)) )
    
    b = mean(ys) - (m * mean(xs))

    return m, b


def squared_error(ys_orig, ys_line):
    """
    Compute the total squared error between the original and predicted y-values.

    This function calculates the sum of the squares of the differences between the actual y-values
    and the predicted y-values on the regression line, providing a measure of the accuracy of the
    regression model.
    """

    return sum((ys_line - ys_orig)**2)


def coefficient_of_determination(ys_orig, ys_line):
    """
    Calculate the coefficient of determination (R^2), a statistical measure of how well the regression model fits the observed data.

    R^2 quantifies the proportion of the variance in the dependent variable that is predictable from the independent variables. 
    It ranges from 0 (no explanatory power) to 1 (perfect fit), where a higher value indicates a model that better explains 
    the variability of the response data around its mean.
    """

    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


# -------- Dataset Creation --------
# POSITIVE CORRELATION
# Data points are spread apart with a higher variance and the R^2 is a lot weaker
# xs, ys = create_dataset(40, 40, 2, correlation="positive")

# POSITIVE CORRELATION
# Data points are closer with a lower variance and the R^2 is much stronger
xs, ys = create_dataset(40, 10, 2, correlation="positive")

# NEGATIVE CORRELATION
# xs, ys = create_dataset(40, 80, 2, correlation="negative")

# NO CORRELATION
# xs, ys = create_dataset(40, 80, 2, correlation=False) 
# ----------------------------------

m, b = best_fit_slope_and_intercept(xs, ys)

# y = mx + b is done here
regression_line = [(m * x) + b for x in xs]
# print(regression_line)

predict_x = 8
predict_y = (m * predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)
print("R Squared Measurement: ", r_squared)

plt.scatter(xs, ys, label="Data")
plt.scatter(predict_x, predict_y, color="g", s=70, label="Prediction")
plt.plot(xs, regression_line, color="red", linewidth=1.2, label="Regression Line")
plt.legend(loc="lower right")
plt.show()