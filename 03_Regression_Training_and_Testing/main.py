import pandas as pd
import nasdaqdatalink
import math
import numpy as np


from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm


pd.set_option("display.max_columns", None) 
nasdaqdatalink.ApiConfig.api_key = ""

df = nasdaqdatalink.get("WIKI/GOOGL")
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

df["HL_PCT"] = (df["Adj. High"] - df["Adj. Low"]) / df["Adj. Close"] * 100.0
df["PCT_CHANGE"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0
df = df[["Adj. Close", "HL_PCT", "PCT_CHANGE", "Adj. Volume"]]

forecast_col = "Adj. Close"
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))

df["Label"] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)
# X = X[:-forecast_out + 1] dropna does this

# features - everything except the label column
X = np.array(df.drop(["Label"], axis=1))
X = preprocessing.scale(X)

# label
y = np.array(df["Label"])

# print(len(X)) # these should match
# print(len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 20% of the data we want for training


# clf = LinearRegression(n_jobs=-1) # use as many threads as available
clf = LinearRegression(n_jobs=1) 
clf.fit(X_train, y_train) # fit = train
accuracy = clf.score(X_test, y_test) # score = test (you can also use the word "confidence")

print(accuracy) # predicting what the price would be shifted 1% for the day

# Loop method to try different kernels
# for k in ['linear','poly','rbf','sigmoid']:
    # clf = svm.SVR(kernel=k)
#     clf.fit(X_train, y_train)
#     confidence = clf.score(X_test, y_test)
#     print(k,confidence)
