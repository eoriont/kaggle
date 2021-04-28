import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os


# Read in the data from the disk
df = pd.read_csv(os.getcwd() + '/quiz2-5/StudentsPerformance.csv')

# Test Preparation Course
unique_testprep = list(df["test preparation course"].unique())
df["no testprep"] = df["test preparation course"].apply(lambda entry: 1 if entry == "none" else 0)
df["testprep"] = df["test preparation course"].apply(lambda entry: 1 if entry == "completed" else 0)

# Parent Education
unique_pared = list(df["parental level of education"].unique())
df["bachelor"] = df["parental level of education"].apply(lambda entry: 1 if entry == "bachelor's degree" else 0)
df["some college"] = df["parental level of education"].apply(lambda entry: 1 if entry == "some college" else 0)
df["master"] = df["parental level of education"].apply(lambda entry: 1 if entry == "master's degree" else 0)
df["associate"] = df["parental level of education"].apply(lambda entry: 1 if entry == "associate's degree" else 0)
df["high school"] = df["parental level of education"].apply(lambda entry: 1 if entry == "high school" else 0)
df["some high school"] = df["parental level of education"].apply(lambda entry: 1 if entry == "some high school" else 0)

def train_regressor(df):
    arr = np.array(df)
    y_arr = arr[:,0]
    X_arr = arr[:,1:]

    regressor = LinearRegression()
    regressor.fit(X_arr, y_arr)

    return regressor

if __name__ == "__main__":
    print("Problem A Student Scores \n", df["math score"][-3:])

    print("Problem B", df['math score'].mean())

    no_test_prep = df['test preparation course'].apply(lambda prep: prep == "none")
    print("Problem C", df["math score"][no_test_prep].mean())

    print("Problem D", len(unique_pared))

    cols_to_keep = ["math score", "no testprep", "testprep", "bachelor", "some college", "master", "associate", "high school", "some high school"]
    df = df[cols_to_keep]
    reg = train_regressor(df[:-3])

    print("Problem E", reg.predict(df[cols_to_keep[1:]][-3:]))
