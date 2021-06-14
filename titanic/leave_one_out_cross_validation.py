import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
import os
import matplotlib.pyplot as plt


# Read in the data from the disk
df = pd.read_csv(os.getcwd() + "/titanic/processed_data.csv")

# Filter only the columns we want to work with
keep_cols = ["Survived", "Sex", "Pclass", "Fare", "Age", "SibSp"]
df = df[keep_cols][:100]

def get_accuracies():
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
            knn_model.fit(df_copy[["Sex", "Pclass", "Fare", "Age", "SibSp"]], df_copy["Survived"])

            # Make prediction and test accuracy
            prediction = knn_model.predict(leave_out_row[["Sex", "Pclass", "Fare", "Age", "SibSp"]])
            num_correct += metrics.accuracy_score(leave_out_row["Survived"], prediction)


        accuracies.append(num_correct/len(df))

    return accuracies

# accuracies = get_accuracies()
# print(accuracies)
accuracies = [0.59, 0.64, 0.6, 0.65, 0.64, 0.64, 0.62, 0.56, 0.56, 0.54, 0.51, 0.56, 0.53, 0.54, 0.5, 0.55, 0.5, 0.56, 0.52, 0.56, 0.53, 0.59, 0.53, 0.57, 0.55, 0.59, 0.56, 0.57, 0.55, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59]
indices = [1,3,5,10,15,20,30,40,50,75]
new_accuracies = [a for i, a in enumerate(accuracies, 1) if i in indices]

# plt.plot(np.arange(1, len(df), 1), accuracies)
plt.plot(indices, new_accuracies)
plt.savefig("leave_one_out")
