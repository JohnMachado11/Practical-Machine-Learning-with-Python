import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import preprocessing, neighbors, svm
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import pandas as pd


pd.set_option("display.max_columns", None) 
np.set_printoptions(edgeitems=30)
# pd.set_option("display.max_rows", None)


df = pd.read_csv("breast-cancer-wisconsin.data")
df.replace("?", -99999, inplace=True)
df.drop(["id"], axis=1, inplace=True)

X = np.array(df.drop(["class"], axis=1))
y = np.array(df["class"])

# Convert all of the "bare_nuclei" column to an int datatype
X[:, 5] = X[:, 5].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC(kernel="linear")
# clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print("Accuracy: ", accuracy)

# Access the support vectors and their details
support_vectors = clf.support_vectors_
support_indices = clf.support_
n_support = clf.n_support_

print(f"Number of Support Vectors: {len(support_vectors)}")
print(f"Support Vector Indices: {support_indices}")
print(f"Support Vectors per Class: {n_support}")

# Check the classes and the number of support vectors per class
classes = clf.classes_

# Print class mapping and the number of support vectors per class
for i, class_label in enumerate(classes):
    print(f"Class {class_label}: {n_support[i]} support vectors")

# Generate predictions from the classifier
y_pred = clf.predict(X_test)

# Create and plot the confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Benign", "Malignant"], cmap=plt.cm.Blues)
plt.title("Confusion Matrix for SVM Classifier")
plt.show()