from sklearn.cluster import KMeans
import pandas as pd
import os
import numpy as np

# Read in the data from the disk
original_df = pd.read_csv(os.getcwd() + "/titanic/processed_data.csv")

# Filter only the columns we want to work with
keep_cols = ["Sex", "Pclass", "Fare", "Age", "SibSp"]
df = original_df[keep_cols]

# Min-max normalization
for column in df.columns:
    df[column] = df[column].apply(lambda x: (x - df[column].min())/(df[column].max() - df[column].min()))

# Fit KMeans model with elbow determined in plot
kmeans = KMeans(n_clusters=4).fit(df)

# Include other columns
df["cluster"] = kmeans.labels_
df["Survived"] = original_df["Survived"]

# Print column averages
col_avgs = df.groupby(["cluster"]).mean()
col_avgs["count"] = np.unique(kmeans.labels_, return_counts=True)[1]

print(col_avgs)
