from math import sqrt
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter


style.use("fivethirtyeight")



dataset = {
    "k": [[1, 2], [2, 3], [3, 1]], # features
    "r": [[6, 5], [7, 7], [8, 6]] # label
}

new_features = [5, 7]


# for i in dataset:
    # for ii in dataset[i]:
        # plt.scatter(ii[0], ii[1], s=100, color=i)

# Plotting some data -  1 liner
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100)
plt.show()


def k_nearest_neighbors(data, predict, k=3):

    if len(data) >= k:
        warnings.warn("K is set to a value less than total voting groups!")
    
    # KNN Algos
    return vote_result