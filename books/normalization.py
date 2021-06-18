import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import os
import matplotlib.pyplot as plt

def get_accuracies(df, independent_cols, dependent_col):
    accuracies = []

    # Go through each k value to find accuracy
    for k_value in range(1, len(df)):
        num_correct = 0

        # Leave each value out individually and count accuracy
        for leave_out_index in range(len(df)):
            # Leave row out
            df_copy = df.copy()
            leave_out_row = df_copy.loc[[leave_out_index]]
            df_copy.drop(leave_out_index, inplace=True)

            # Fit KNN model
            knn_model = KNeighborsClassifier(k_value)
            knn_model.fit(df_copy[independent_cols], df_copy[dependent_col])

            # Make prediction and test accuracy
            prediction = knn_model.predict(leave_out_row[independent_cols])
            num_correct += metrics.accuracy_score(leave_out_row[dependent_col], prediction)

        accuracies.append(num_correct/len(df))

    return accuracies

# Generate plot
def generate_plot(accuracies, label):
    indices = list(range(1, 100, 2))
    new_accuracies = [a for i, a in enumerate(accuracies, 1) if i in indices]
    plt.plot(indices, new_accuracies, label=label)

# Read in the data from the disk
df = pd.read_csv(os.getcwd() + "/books/dataset.csv")

# Filter only the columns we want to work with
independent_cols = ["num pages", "num unique words", "avg sentence length", "avg word size"]
dependent_col = "book type"
df = df[independent_cols+[dependent_col]]

# Change the "book type column" to be numeric
df["book type"] = df["book type"].apply(lambda x: 1 if x == "adult book" else 0)

# Plot line for default data
print("Finished unnormalized")
accuracies = get_accuracies(df, independent_cols, dependent_col)
generate_plot(accuracies, "unnormalized")
print("Finished accuracies")

# Plot line for simple scaling normalized data
simple_df = df.copy()
for column in simple_df.columns:
    if column == dependent_col:
        continue
    simple_df[column] /= simple_df[column].max()
print("Finished simple scaling")
simple_accuracies = get_accuracies(simple_df, independent_cols, dependent_col)
generate_plot(simple_accuracies, "simple scaling")
print("Finished accuracies")

# Plot line for min-max normalized data
minmax_df = df.copy()
for column in minmax_df.columns:
    if column == dependent_col:
        continue
    minmax_df[column] = minmax_df[column].apply(lambda x: (x - minmax_df[column].min())/(minmax_df[column].max() - minmax_df[column].min()))
print("Finished min-max")
minmax_accuracies = get_accuracies(minmax_df, independent_cols, dependent_col)
generate_plot(minmax_accuracies, "min-max")
print("Finished accuracies")
#
# Plot line for zscore normalized data
zscore_df = df.copy()
for column in zscore_df.columns:
    if column == dependent_col:
        continue
    zscore_df[column] = zscore_df[column].apply(lambda x: (x - zscore_df[column].mean())/zscore_df[column].std())
print("Finished z-scoring")
zscore_accuracies = get_accuracies(zscore_df, independent_cols, dependent_col)
generate_plot(zscore_accuracies, "z-scoring")
print("Finished accuracies")

# Save plot as an image
plt.legend(loc="lower left")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Leave-One-Out Accuracy for Various Normalizations")
plt.savefig("a121_a")
