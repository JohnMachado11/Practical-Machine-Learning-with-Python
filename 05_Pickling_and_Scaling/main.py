import pandas as pd
import nasdaqdatalink
import math
import numpy as np
import datetime

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use("ggplot")


pd.set_option("display.max_columns", None) 
nasdaqdatalink.ApiConfig.api_key = ""

df = nasdaqdatalink.get("WIKI/GOOGL")

df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Low"]) / df["Adj. Close"] * 100.0
df["PCT_CHANGE"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0

#             These directly impact the price
#          price         x            x             x
df = df[["Adj. Close", "HL_PCT", "PCT_CHANGE", "Adj. Volume"]]
forecast_col = "Adj. Close"
df.fillna(-99999, inplace=True)

# 1%
# forecast_out = int(math.ceil(0.01 * len(df)))

# 10%
forecast_out = int(math.ceil(0.1 * len(df)))
df["Label"] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(["Label"], axis=1))
X = preprocessing.scale(X)

# Have X_Lately come first in variable assignment so it grabs the last 10% of X. 
# This is so it doesn't grab the last 10% of the X which comes after X_Lately.
# The X after X_Lately is the first 90% of X. So if I flipped these variables around, 
# X_Lately would take the last 10% of the 90% of the new X. 
# The order is important here.
X_Lately = X[-forecast_out:] # last 35 days
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df["Label"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# clf = LinearRegression(n_jobs=1)
# clf.fit(X_train, y_train)
# print("Training Model")
# with open("linear_regression.pickle", "wb") as f:
#     pickle.dump(clf, f)

# print("Loading new model")
pickle_in = open("linear_regression.pickle", "rb")
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_Lately)
print(forecast_set, accuracy, forecast_out)
df["Forecast"] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86_400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
    # print(df.loc[next_date])

print(df.tail())

df["Adj. Close"].plot()
df["Forecast"].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()