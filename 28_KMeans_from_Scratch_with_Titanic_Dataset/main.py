import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing
import pandas as pd

style.use("ggplot")
pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
np.set_printoptions(threshold=np.inf,linewidth=np.inf)


df = pd.read_excel("titanic.xls")
df.drop(["body", "name"], axis=1, inplace=True)
df.apply(pd.to_numeric, errors="ignore")
df.fillna(0, inplace=True)


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
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            
            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]

                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                    optimized = False
            
            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


def handle_non_numerical_data(df):

    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0

            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            
            df[column] = list(map(convert_to_int, df[column]))

    return df


df = handle_non_numerical_data(df)

# Dropping "ticket" actually makes it somewhat worse the accuracy.
# Most likely due to related family members having the same ticket #.
# df.drop(["ticket"], axis=1, inplace=True)

df.drop(["boat"], axis=1, inplace=True)
# df.drop(["sex"], axis=1, inplace=True)

X = np.array(df.drop(["survived"], axis=1).astype(float))
X = preprocessing.scale(X)
y = np.array(df["survived"])

# print(X[:5])

clf = KMeans()
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)

    if prediction == y[i]:
        correct += 1


print("Correct: ", correct)
print(correct / len(X))