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

style.use("ggplot")


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


X = np.array(df.drop(["Label"], axis=1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_Lately = X[-forecast_out:] # last 35 days

df.dropna(inplace=True)
y = np.array(df["Label"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

# print(accuracy)

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