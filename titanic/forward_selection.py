from logistic_analysis import get_regressor_accuracy, train_regressor
from interaction_analysis import get_combos, apply_combos
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
import os

from logistic_analysis import df
# df = pd.read_csv(os.getcwd() + "/titanic/processed_data.csv")
combos = get_combos(df)
df = apply_combos(df, combos)

possible_cols = list(df.columns)
possible_cols.remove("Survived")

cols = []
last_accuracy = ("", 0)
while True:
    best_accuracy = ("", 0)

    for col in possible_cols:
        new_df = df[["Survived", col]+cols]
        reg = train_regressor(new_df[500:])
        accuracy = (col, get_regressor_accuracy(new_df, reg))
        if accuracy[1] > best_accuracy[1]:
            best_accuracy = accuracy

    if best_accuracy[1] < last_accuracy[1]:
        break
    else:
        cols.append(best_accuracy[0])
        last_accuracy = best_accuracy
        possible_cols.remove(best_accuracy[0])

    if len(possible_cols) == 0:
        break



reg = train_regressor(df[:500])
train = get_regressor_accuracy(df[:500], reg)
test = get_regressor_accuracy(df[500:], reg)
print('\n')
print('Features:', cols)
print('')
print('Training Accuracy:', train)
print('Testing Accuracy:', test)
print('')
