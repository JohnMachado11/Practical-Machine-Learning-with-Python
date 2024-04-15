from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use("fivethirtyeight")


xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


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

m, b = best_fit_slope_and_intercept(xs, ys)


# y = mx + b is done here
regression_line = [(m * x) + b for x in xs]
print(regression_line)

predict_x = 8
predict_y = (m * predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)

plt.scatter(xs, ys, label="Data")
plt.scatter(predict_x, predict_y, color="g", label="Prediction")
plt.plot(xs, regression_line, color="red", label="Regression Line")
plt.legend(loc="lower right")
plt.show()