from logistic_analysis import df, get_regressor_accuracy, train_regressor
from interaction_analysis import get_combos
import pandas as pd
from sklearn.linear_model import LogisticRegression

possible_cols = get_combos(df) + list(df.columns)

while True:
    best_accuracy = ("", 0)

    for col in possible_cols:
        r = train_regressor()
