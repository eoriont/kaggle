from logistic_analysis import df, get_regressor_accuracy
import re

def get_combos(df):
    combos = []
    for col in df.columns:
        for col2 in df.columns:
            if col != col2 \
                and "Survived" not in (col, col2) \
                and (col2, col) not in combos \
                and re.split('=|>| ', col)[0] != re.split('=|>| ', col2)[0]:
                combos.append((col, col2))
    return combos

def apply_combos(df, combos):
    for c1, c2 in combos:
        df[c1 + "*" + c2] = df[c1] * df[c2]
    return df

if __name__ == "__main__":
    get_regressor_accuracy(apply_combos(df, get_combos(df)))
