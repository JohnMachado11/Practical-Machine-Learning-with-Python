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


m, b = best_fit_slope_and_intercept(xs, ys)

print("m: ", m)
print("b: ", b)

regression_line = [(m * x) + b for x in xs]
print(regression_line)

predict_x = 8
predict_y = (m * predict_x) + b

plt.scatter(xs, ys, label="Data")
plt.scatter(predict_x, predict_y, color="g", label="Prediction")
plt.plot(xs, regression_line, color="red", label="Regression Line")
plt.legend(loc="lower right")
plt.show()