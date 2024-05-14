from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np


style.use("ggplot")

X = np.array([
            [1, 2],
            [1.5, 1.8],
            [5, 8],
            [8, 8],
            [1, 0.6],
            [9, 11]])

# print(X)

# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()

clf = KMeans(n_clusters=2)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

print(centroids)
print(labels)

# Each element in the labels array corresponds to a data point in X and indicates which cluster the data point belongs to. 
# For example, if labels[0] is 1, it means the first data point [1, 2] is assigned to cluster 1.

colors = 10 * ["g.", "r.", "c.", "b.", "k."]

# Plot each data point with its corresponding cluster color
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=25)


# Plot the centroids of the clusters
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5)
plt.show()