from logistic_analysis import get_regressor_accuracy, train_regressor
from interaction_analysis import get_combos, apply_combos
import pandas as pd

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from sklearn.linear_model import LogisticRegression
import os

from logistic_analysis import df
df = pd.read_csv(os.getcwd() + "/titanic/processed_data.csv")
# combos = get_combos(df)
# df = apply_combos(df, combos)

possible_cols = list(df.columns)
possible_cols.remove("Survived")
possible_cols.remove("id")

total_df = df[["Survived"] + possible_cols]
total_regressor = train_regressor(total_df[:500], max_iter=10)
total_accuracy = get_regressor_accuracy(total_df[500:], total_regressor)
print(total_accuracy)
print(list(total_df.columns))

remove_col_indices = []
for j, col in enumerate(possible_cols):
    cols = [x for i, x in enumerate(possible_cols) if i not in remove_col_indices]
    new_df = df[["Survived"]+cols]
    reg = train_regressor(new_df[:500], max_iter=10)
    accuracy = get_regressor_accuracy(new_df[500:], reg)

    if accuracy <= total_accuracy:
        remove_col_indices.append(j)
    else:
        total_accuracy = accuracy

total_cols = [x for i, x in enumerate(possible_cols) if i not in remove_col_indices]
total_df = df[["Survived"]+cols]
reg = train_regressor(total_df[:500])
train = get_regressor_accuracy(total_df[:500], reg)
test = get_regressor_accuracy(total_df[500:], reg)
print(remove_col_indices)
print('Features:', cols)
print('')
print('Training Accuracy:', train)
print('Testing Accuracy:', test)
print('')
