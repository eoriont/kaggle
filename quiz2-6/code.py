import pandas as pd
import os, re


# Read in the data from the disk
df = pd.read_csv(os.getcwd() + '/quiz2-6/data.csv')

df["city_id"] = df["city"].apply(lambda x: int(x.split("_")[1]))

# Problems
print("Problem A", df["training_hours"].mean())

print("Problem B", len(df[df['target'].apply(lambda x: float(x) == 1)])/len(df))

print("Problem C", df["city"].mode()[0])

print("Problem D", df["city"].value_counts()[0])

print("Problem E", "city_" + str(df["city_id"].max()))

df["min_company_sizes"] = df["company_size"].apply(lambda x: 0 if pd.isnull(x) else float([y for y in re.split(r"\D+", x) if y != ""][-1]))
print("Problem F", len(df[df["min_company_sizes"].apply(lambda x: x < 10)]))

print("Problem G", len(df[df["min_company_sizes"].apply(lambda x: x < 100)]))
