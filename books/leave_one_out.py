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

# Read in the data from the disk
df = pd.read_csv(os.getcwd() + "/books/dataset.csv")

# Filter only the columns we want to work with
independent_cols = ["num pages", "num unique words", "avg sentence length", "avg word size"]
dependent_col = "book type"
df = df[independent_cols+[dependent_col]]



accuracies = get_accuracies(df, independent_cols, dependent_col)
print(accuracies)

# Generate plot
indices = list(range(1, 100, 2))
new_accuracies = [a for i, a in enumerate(accuracies, 1) if i in indices]

plt.plot(indices, new_accuracies)
plt.savefig("a121_a")
