import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import preprocessing
import pandas as pd

style.use("ggplot")
pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)


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

print(df.head())