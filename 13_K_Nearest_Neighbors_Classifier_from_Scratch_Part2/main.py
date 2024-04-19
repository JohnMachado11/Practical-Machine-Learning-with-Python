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


def k_nearest_neighbors(data, predict, k=3):
    """
    Predict the class of a given data point using the k-Nearest Neighbors algorithm.

    This function calculates the Euclidean distance between a given test point and all points in the training set. 
    It selects the k closest points and uses a majority vote among their classes to predict the class of the test point.
    """

    if len(data) >= k:
        warnings.warn("K is set to a value less than total voting groups!")

    distances = []
    for group in data:
        for features in data[group]:
            # Fastest
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
            # Alternate
            # np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
    
    # print(sorted(distances)[:k])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result


result = k_nearest_neighbors(dataset, new_features, k=3)
print("Result: ", result)


[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], color=result, s=100)
plt.show()