from math import sqrt
import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random


pd.set_option("display.max_columns", None) 
np.set_printoptions(edgeitems=30)



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
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence

accuracies = []

for i in range(25):
    df = pd.read_csv("breast-cancer-wisconsin.data")
    df.replace("?", -99999, inplace=True)
    df.drop(["id"], axis=1, inplace=True)
    full_data = df.astype(float).values.tolist()

    random.shuffle(full_data)

    # test_size = 0.2
    test_size = 0.4
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}

    # Everything up to the last 20% of data
    train_data = full_data[:-int(test_size * len(full_data))]

    # The last 20% of data
    test_data = full_data[-int(test_size * len(full_data)):]

    for i in train_data:
        # Append to list within the "2" or "4" key in the "train_set" dictionary.
        # Append all the data of a particular list of data except the very last element
        # which is either "2" or "4", which is the Label, 2 = "benign" 4 = "malignant".
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])


    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            # else:
            #     print(confidence)

            total += 1

    print("Accuracy: ", correct / total)
    accuracies.append(correct / total)

print(sum(accuracies) / len(accuracies))