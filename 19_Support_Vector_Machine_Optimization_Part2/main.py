import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use("ggplot")


class SupportVectorMachine:
    
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: "r", -1: "b"}
        
        if self.visualization: 
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        self.data = data
        
        # {  ||w||: [w, b] }
        opt_dict = {}

        transforms = [[1, 1],
                    [-1, 1],
                    [-1, -1],
                    [1, -1]]
        
        all_data = []

        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # support vectors yi(xi . w+b) = 1
        # You will know that you have found a really great value for w and b 
        # when both your positive and negative classes have a value that is close to 1.
        # How close to 1 is up to you. You can determine this. 
        # "I want to be at least > 2"
        # or
        # "1.01 or better"

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                    # point of expense:
                      self.max_feature_value * 0.001]
        

        # Extremely expensive
        b_range_multiple = 5
        
        # we dont need to take as small of steps
        # with b as we do w
        b_multiple = 5

        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            # starting at the top of the bowl 'U',  'dropping the ball here'
            w = np.array([latest_optimum, latest_optimum])
            
            # We can do this because Convex
            optimized = False

            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                    self.max_feature_value * b_range_multiple,
                                    step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        # Weakest link in the SVM fundamentally
                        # SMO (Sequential Minimal Optimization) attemps to fix this a bit
                        # yi(xi . w+b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    # break could go here if its False
                        
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print("Optimized a step.")
                else:
                    # w = [5, 5]
                    # step = 1
                    # w - step = [4, 4] (or [5, 5] - [step, step])
                    w = w - step
            
            # magnitudes sorted
            norms = sorted([n for n in opt_dict])
            
            # ||w||: [w, b]
            opt_choice = opt_dict[norms[0]] # first element
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2


    def predict(self, features):
        # sign ( x.w + b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        return classification


data_dict = {
    -1: np.array([
            [1, 7],
            [2, 8],
            [3, 8]]),

    1: np.array([
        [5, 1],
        [6, -1],
        [7, 3]])
    }
