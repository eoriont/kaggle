from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import os

# Read in the data from the disk
df = pd.read_csv(os.getcwd() + "/titanic/processed_data.csv")

# Filter only the columns we want to work with
keep_cols = ["Sex", "Pclass", "Fare", "Age", "SibSp"]
df = df[keep_cols]

# Min-max normalization
for column in df.columns:
    df[column] = df[column].apply(lambda x: (x - df[column].min())/(df[column].max() - df[column].min()))

# Generate the plot
k_range = list(range(1, 26))
sum_squared_errors = []
for k_value in k_range:
    # Fit kmeans model
    kmeans = KMeans(n_clusters=k_value)
    kmeans.fit(df)

    sum_squared_errors.append(kmeans.inertia_)

plt.xlabel("k")
plt.ylabel("sum squared distances from cluster centers")
plt.title("K-Means Clustering on Titanic Dataset")
plt.xticks(k_range)
plt.plot(k_range, sum_squared_errors)
plt.savefig("a129")
