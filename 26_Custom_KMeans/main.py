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

plt.scatter(X[:, 0], X[:, 1], s=75)
plt.show()

colors = 10 * ["g", "r", "c", "b", "k"]

class KMeans:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]
        
        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []
            
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                pass
                #self.centroids[classification] = np.average(self.classifications[classification], axis=0)


    def predict(self, data):
        pass

