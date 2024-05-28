import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import preprocessing
import pandas as pd

style.use("ggplot")
pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
np.set_printoptions(threshold=np.inf,linewidth=np.inf)


df = pd.read_excel("titanic.xls")
original_df = pd.DataFrame.copy(df)

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

df.drop(["ticket", "home.dest"], axis=1, inplace=True)

X = np.array(df.drop(["survived"], axis=1).astype(float))
X = preprocessing.scale(X)
y = np.array(df["survived"])

# print(X[:5])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df["cluster_group"] = np.nan

# Indicate which cluster a particular row of data belongs to
for i in range(len(X)):
    original_df["cluster_group"].iloc[i] = labels[i]


n_clusters_ = len(np.unique(labels))
survival_rates = {}

for i in range(n_clusters_):
    # If "i" == 0, then this temp df is the original df only where the cluster group is 0
    temp_df = original_df[ (original_df["cluster_group"] == float(i)) ]

    survival_cluster = temp_df[ (temp_df["survived"] == 1) ]
    survival_rate = len(survival_cluster) / len(temp_df)

    survival_rates[i] = survival_rate

print(survival_rates)

# print(original_df[(original_df["cluster_group"] == 0)].head())
# print(original_df[(original_df["cluster_group"] == 0)].describe())

cluster_0 = (original_df[ (original_df['cluster_group'] == 0) ])
cluster_0_fc = (cluster_0[ (cluster_0['pclass'] == 1) ])

# Survival rate of the 1st class passengers in cluster 0, 
# compared to the overall survival rate of cluster 0
print(cluster_0_fc.describe())