from sklearn.model_selection import cross_validate, train_test_split
from sklearn import preprocessing, neighbors, svm
import numpy as np
import pandas as pd


pd.set_option("display.max_columns", None) 
np.set_printoptions(edgeitems=30)
# pd.set_option("display.max_rows", None)


df = pd.read_csv("breast-cancer-wisconsin.data")
df.replace("?", -99999, inplace=True)
df.drop(["id"], axis=1, inplace=True)

# print(df.head())

X = np.array(df.drop(["class"], axis=1))
y = np.array(df["class"])

# Convert all of the "bare_nuclei" column to an int datatype
X[:, 5] = X[:, 5].astype(int)

# print(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC(kernel="linear")
# clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print("Accuracy: ", accuracy)

# ----------------------------------------------------------
# Benign example
# example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1]])

# Maligant example
# example_measures = np.array([[8, 10, 10, 1, 1, 2, 3, 2, 1]])
# ----------------------------------------------------------

# ----------------------------------------------------------
# Reshaping + multiple samples
# More than 1 sample
example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [8, 10, 10, 1, 1, 2, 3, 2, 1]])

# Reshape is redundant in my case right now.
# reshape() is saying, make 2 rows with 9 columns per row. 
#      1. len(example_measures) = 2
#      2. -1 = 9
# The data within "example_measures" is already in that exact shape, 2 rows and 9 columns per row.
# reshape() is just another way to do that transformation.

# example_measures = example_measures.reshape(len(example_measures), -1)
# ----------------------------------------------------------

prediction = clf.predict(example_measures)

print()
# 1 sample
if len(prediction) == 1:
    if prediction[0] == 2:
        print("Benign Prediction: ", prediction[0])
    else:
        print("Maligant Prediction: ", prediction[0])
else: 
    # More than 1 sample
    for i in prediction:
        if i == 2:
            print("Benign Prediction: ", i)
        else:
            print("Maligant Prediction: ", i)

