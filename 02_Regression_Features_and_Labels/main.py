import pandas as pd
import nasdaqdatalink
import math


pd.set_option("display.max_columns", None)  
nasdaqdatalink.ApiConfig.api_key = ""

df = nasdaqdatalink.get("WIKI/GOOGL")
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

df["HL_PCT"] = (df["Adj. High"] - df["Adj. Low"]) / df["Adj. Close"] * 100.0
df["PCT_CHANGE"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0

df = df[["Adj. Close", "HL_PCT", "PCT_CHANGE", "Adj. Volume"]]

forecast_col = "Adj. Close"
df.fillna(-99999, inplace=True)

# Generally you use Regression to forecast OUT

# math.ceil() will take anything and go to the ceiling.
# If the length of the dataframe was something like .2, math.ceil() will round that up to 1, rounds up to the nearest whole. 
# We want to predict up to 1% of the dataframe.
forecast_out = int(math.ceil(0.01 * len(df)))

df["Label"] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)