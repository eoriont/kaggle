import multiprocessing
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

# Constants
k_values = list(range(1, 100, 2))
independent_cols = ["Sex", "Pclass", "Fare", "Age", "SibSp"]
dependent_col = "Survived"
multiprocessing = True

def get_accuracies(mat):
    accuracies = []

    independent_mat = mat[:, :-1]
    dependent_mat = mat[:, -1]

    # Go through each k value to find accuracy
    # for k_value in range(1, len(mat)):
    for k_value in k_values:
        knn_model = KNeighborsClassifier(k_value)
        num_correct = 0

        # Leave each value out individually and count accuracy
        for leave_out_index in range(len(mat)):

            # Leave row out
            indices = np.arange(len(mat))
            test_x = independent_mat[leave_out_index, :]
            test_y = dependent_mat[leave_out_index]
            train_x = independent_mat[indices != leave_out_index, :]
            train_y = dependent_mat[indices != leave_out_index]

            # Fit KNN model
            knn_model.fit(train_x, train_y)

            # Make prediction and test accuracy
            prediction = knn_model.predict([test_x])
            if prediction[0] == test_y:
                num_correct += 1

        accuracies.append(num_correct/len(mat))
    return accuracies

def normalize_data(mat, method):
    if method == "none":
        return mat
    elif method == "simple":
        # Split off dependent mat, normalize, then add independent, then return
        # dm = Dependent Matrix
        dm = mat[:, :-1]
        m = dm / dm.max(axis=0)
        return np.append(m, np.reshape(mat[:, -1], (len(m), 1)), axis=1)
    elif method == "minmax":
        # Split off dependent mat, normalize, then add independent, then return
        dm = mat[:, :-1]
        m = (dm - dm.min(axis=0))/(dm.max(axis=0) - dm.min(axis=0))
        return np.append(m, np.reshape(mat[:, -1], (len(m), 1)), axis=1)
    elif method == "zscore":
        # Split off dependent mat, normalize, then add independent, then return
        dm = mat[:, :-1]
        m = (dm - dm.mean(axis=0))/(dm.std(axis=0))
        return np.append(m, np.reshape(mat[:, -1], (len(m), 1)), axis=1)

# Generate plot
def generate_plot(accuracies, label):
    indices = list(range(1, 100, 2))
    # new_accuracies = [a for i, a in enumerate(accuracies, 1) if i in indices]
    new_accuracies = accuracies
    plt.plot(indices, new_accuracies, label=label)

if __name__ == "__main__":
    start_time = time.time()

    # Start threadpool
    pool = Pool(4)

    # Read in the data from the disk
    df = pd.read_csv(os.getcwd() + "/titanic/processed_data.csv")

    # Filter only the columns and rows we want to work with
    # The columns have to be in this order because we split it up later in a certain way
    df = df[:100][independent_cols+[dependent_col]]
    mat = np.array(df)
    normalization_types = ["none", "simple", "minmax", "zscore"]
    mats = [np.copy(mat) for _ in normalization_types]

    if multiprocessing:
        # Multiprocessing way (the cool way)
        print("Normalizing the data using multiprocessing...")
        mat_array = pool.starmap(normalize_data, zip(mats, normalization_types))
    else:
        # Non multiprocessing way
        print("Normalizing data...")
        mat_array = [normalize_data(m, method) for m, method in zip(mats, normalization_types)]

    array_order = ["unnormalized", "simple scaling", "min-max", "z-scoring"]


    if multiprocessing:
        # Multiprocessing way (the cool way)
        print("Multiprocessing accuracies...")
        results = pool.map(get_accuracies, mat_array)
    else:
        # Non multiprocessing way
        print("Getting accuracies...")
        results = [get_accuracies(mat) for mat in mat_array]

    # Plot the accuracies
    for accuracies, name in zip(results, array_order):
        generate_plot(accuracies, name)
    print("Finished accuracies")

    # Save plot as an image
    plt.legend(loc="upper right")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Leave-One-Out Accuracy for Various Normalizations")
    plt.savefig("a122_a")
    print("Total Time:", time.time()-start_time)
