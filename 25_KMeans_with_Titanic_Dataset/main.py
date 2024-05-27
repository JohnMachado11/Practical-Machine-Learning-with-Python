import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import preprocessing
import pandas as pd
from sklearn.decomposition import PCA

style.use("ggplot")
pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
np.set_printoptions(threshold=np.inf,linewidth=np.inf)


"""
            ---------------------------------------------
                    ---- COLUMN EXPLANATIONS ----        
            ---------------------------------------------

Pclass                                                 survival
Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)            Survival (0 = No; 1 = Yes)

name                                                   sex 
Name                                                   Sex 

age                                                    sibsp  
Age                                                    Number of Siblings/Spouses Aboard

parch                                                  ticket
Number of Parents/Children Aboard                      Ticket Number

fare                                                   cabin
Passenger Fare (British pound)                         Cabin

embarked                                               boat
Port of Embarkation                                    Lifeboat
(C = Cherbourg; Q = Queenstown; S = Southampton)

body                                                   home.dest
Body Identification Number                             Home/Destination
"""

df = pd.read_excel("titanic.xls")

df.drop(["body", "name"], axis=1, inplace=True)


df.apply(pd.to_numeric, errors="ignore")

df.fillna(0, inplace=True)


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

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)

    if prediction[0] == y[i]:
        correct += 1


# print(df.head())

# In this case, survived is either a 0, which means non-survival, or a 1, which means survival. 
# For a clustering algorithm, the machine will find the clusters, but then will asign arbitrary 
# values to them, in the order it finds them. Thus, the group that is survivors might be a 0 or a 1,
# depending on a degree of randomness. Thus, if you consistently get 30% and 70% accuracy, then your model is 70% accurate.
print("Correct: ", correct)
print(correct / len(X))