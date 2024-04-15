import pandas as pd
import nasdaqdatalink


pd.set_option("display.max_columns", None) 
nasdaqdatalink.ApiConfig.api_key = ""

# start = '2004-01-01'  # Example start date

df = nasdaqdatalink.get("WIKI/GOOGL")


df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

# We need to define the relationships for the regression

# High - Low Percent (volatility)
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Low"]) / df["Adj. Close"] * 100.0

# Daily move (New - Old) / Old * 100
df["PCT_CHANGE"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0

df = df[["Adj. Close", "HL_PCT", "PCT_CHANGE", "Adj. Volume"]]

print(df.head())