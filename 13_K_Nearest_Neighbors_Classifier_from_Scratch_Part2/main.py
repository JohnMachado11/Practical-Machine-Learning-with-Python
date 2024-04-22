from math import sqrt
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter


style.use("fivethirtyeight")


dataset = {
    # Features = elements within the lists
    # Labels = "k" and "r"
    "k": [[1, 2], [2, 3], [3, 1]],
    "r": [[6, 5], [7, 7], [8, 6]]
}

# new_features = [5, 7]
new_features = [4, 2]


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
            # ----------------------------------------------------
            #                    Step by Step
            print("Features:", np.array(features))
            print("Predict:", np.array(predict))
            print("Subtraction: ", np.array(features) - np.array(predict))
            print("Squared: ", ((np.array(features) - np.array(predict)))**2)
            print("Summed: ", np.sum(((np.array(features) - np.array(predict))**2)))
            print("Square Root: ", np.sqrt(np.sum((np.array(features) - np.array(predict))**2)))
            # ----------------------------------------------------

            # Fastest
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            print("Distance: ", euclidean_distance)
            distances.append([euclidean_distance, group])
            # Alternate
            # np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
            print("-------\n")
    
    print(distances)
    print(sorted(distances))
    print(sorted(distances)[:k]) # top 3

    votes = [i[1] for i in sorted(distances)[:k]]
    print(votes)
    vote_result = Counter(votes).most_common(1)[0][0]
    print(vote_result)

    return vote_result


result = k_nearest_neighbors(dataset, new_features, k=3)
print("Result: ", result)

#                  Plotting with labels
# for label in dataset:
#     class_points = dataset[label]
#     xs = [point[0] for point in class_points]
#     ys = [point[1] for point in class_points]
#     plt.scatter(xs, ys, s=100, label=label.upper())

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], color=result, s=100)
plt.scatter(new_features[0], new_features[1], color="green", s=100, label="Prediction")
plt.legend(loc="lower right")
plt.show()