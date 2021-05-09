from logistic_analysis import get_regressor_accuracy, train_regressor
import pandas as pd

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

import os

from logistic_analysis import df

def print_accuracies(reg, df):
    h = len(df)//2
    print("Training Accuracy:", get_regressor_accuracy(df[:h], reg))
    print("Testing Accuracy:", get_regressor_accuracy(df[h:], reg))

df = pd.read_csv(os.getcwd() + "/titanic/processed_data.csv")

possible_cols = list(df.columns)
possible_cols.remove("Survived")
possible_cols.remove("id")

total_df = df[["Survived"] + possible_cols]
halfway = 500#len(total_df)//2
total_regressor = train_regressor(total_df[:halfway], max_iter=100)

print("Accuracy Before:")
print_accuracies(total_regressor, total_df)
print("")

cols_to_remove = []
total_accuracy = get_regressor_accuracy(total_df[halfway:], total_regressor)
for col in possible_cols:
    new_df = total_df.drop([*cols_to_remove, col], axis=1)

    new_reg = train_regressor(new_df[:halfway], max_iter=100)
    new_accuracy = get_regressor_accuracy(new_df[halfway:], new_reg)

    # print("Column:", col)
    # print_accuracies(new_reg, new_df)
    # print("removed" if new_accuracy >= total_accuracy else "kept")
    # print("Baseline accuracy:", total_accuracy)

    # Remove column if it becomes more accurate
    if new_accuracy >= total_accuracy:
        total_accuracy = new_accuracy
        cols_to_remove.append(col)

    # print("Removed columns:", cols_to_remove)
    # print("")

total_df = total_df.drop(cols_to_remove, axis=1)
total_regressor = train_regressor(total_df[:halfway], max_iter=100)

print("Backwards Selection Accuracies:")
print_accuracies(total_regressor, total_df)
