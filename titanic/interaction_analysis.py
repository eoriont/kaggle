from logistic_analysis import df, get_df_accuracy
import re

combos = []
for col in df.columns:
    for col2 in df.columns:
        if col != col2 \
            and "Survived" not in (col, col2) \
            and (col2, col) not in combos \
            and re.split('=|>| ', col)[0] != re.split('=|>| ', col2)[0]:
            combos.append((col, col2))

for c1, c2 in combos:
    df[c1 + "*" + c2] = df[c1] * df[c2]

if __name__ == "__main__":
    get_df_accuracy(df)
