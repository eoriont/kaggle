import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os


# Read in the data from the disk
df = pd.read_csv(os.getcwd() + '/titanic/train.csv')

# Filter only the columns we want to work with
keep_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
df = df[keep_cols]

#>==============================
# Processing the columns
#>==============================

# "Sex" column
def convert_sex_to_int(sex):
    if sex == 'male':
        return 0
    elif sex == 'female':
        return 1

df['Sex'] = df['Sex'].apply(convert_sex_to_int)

# "Age" column
age_nan = df['Age'].apply(lambda entry: np.isnan(entry))
age_not_nan = df['Age'].apply(lambda entry: not np.isnan(entry))

# Set each nil column to the mean age
df.loc[age_nan, ['Age']] = df['Age'][age_not_nan].mean()

# SibSp
def indicator_greater_than_zero(x):
    if x > 0:
        return 1
    else:
        return 0

df['SibSp>0'] = df['SibSp'].apply(indicator_greater_than_zero)

# Parch
df['Parch>0'] = df['Parch'].apply(indicator_greater_than_zero)

# CabinType

df['Cabin']= df['Cabin'].fillna('None')

def get_cabin_type(cabin):
    if cabin != 'None':
        return cabin[0]
    else:
        return cabin

df['CabinType'] = df['Cabin'].apply(get_cabin_type)

for cabin_type in df['CabinType'].unique():
    dummy_variable_name = 'CabinType={}'.format(cabin_type)
    dummy_variable_values = df['CabinType'].apply(lambda entry: int(entry == cabin_type))
    df[dummy_variable_name] = dummy_variable_values

del df['CabinType']

# Embarked

df['Embarked'] = df['Embarked'].fillna('None')

for cabin_type in df['Embarked'].unique():
    dummy_variable_name = 'Embarked={}'.format(cabin_type)
    dummy_variable_values = df['Embarked'].apply(lambda entry: int(entry == cabin_type))
    df[dummy_variable_name] = dummy_variable_values

del df['Embarked']

features_to_use = ['Sex', 'Pclass', 'Fare', 'Age', 'SibSp', 'SibSp>0', 'Parch>0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']
columns_needed = ['Survived'] + features_to_use
df = df[columns_needed]

def train_regressor(df):
    arr = np.array(df)
    y_arr = arr[:,0]
    X_arr = arr[:,1:]

    regressor = LogisticRegression(max_iter=1000)
    regressor.fit(X_arr, y_arr)

    coef_dict = {}
    feature_columns = df.columns[1:]
    feature_coefficients = regressor.coef_
    for i in range(len(feature_columns)):
        column = feature_columns[i]

        # This returns an array with the coefficient array inside for some reason
        coefficient = feature_coefficients[0][i]

        coef_dict[column] = coefficient
    return regressor

def get_regressor_accuracy(df, regressor):
    arr = np.array(df)
    y_arr = arr[:,0]
    X_arr = arr[:,1:]

    y_predictions = regressor.predict(X_arr)
    y_predictions = [convert_regressor_output_to_survival_value(output) for output in y_predictions]

    return get_accuracy(y_predictions, y_arr)

def convert_regressor_output_to_survival_value(output):
    if output < 0.5:
        return 0
    else:
        return 1

def get_accuracy(predictions, actual):
    num_correct = 0
    num_incorrect = 0

    for i in range(len(predictions)):
        if predictions[i] == actual[i]:
            num_correct += 1
        else:
            num_incorrect += 1

    return num_correct / (num_correct + num_incorrect)

if __name__ == "__main__":
    # split into training/testing dataframes
    reg = train_regressor(df[:500])
    train = get_regressor_accuracy(df[:500], reg)
    test = get_regressor_accuracy(df[500:], reg)

    print('\n')
    print('features:', len(df.columns))
    print('')
    print('training accuracy:', train)
    print('testing accuracy:', test)
    print('')
