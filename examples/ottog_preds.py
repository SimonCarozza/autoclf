from pandas import read_csv, DataFrame
import os
import sys
from autoclf.encoding import labelenc as lc
from keras.models import load_model
from sklearn.externals import joblib as jl
# import sklearn
import numpy as np
from random import sample

# autoclf classification's module
from autoclf.classification import predict as pr
from pkg_resources import resource_string
from io import StringIO

# Let's make predictions and print their associated probabilities

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load data

ottog_bytes = resource_string(
    "autoclf", os.path.join("datasets", 'otto_group_test.csv'))
ottog_file = StringIO(str(ottog_bytes,'utf-8'))

df = read_csv(ottog_file, delimiter=",")

df.drop(['id'], axis=1, inplace=True)

print("shape: ", df.shape)

print()

# description = df.describe()
# print("Once again, description - no encoding:\n", description)

if df.isnull().values.any():
    print("df has null values")
    df.fillna(method='pad', inplace=True)
    df.fillna(method='bfill', inplace=True)

print()
# input("Enter key to continue... \n")

original_df = df

original_X = original_df.values

print("Original data sample before encoding (values):\n", original_X[:2])

df = lc.dummy_encode(df)
df.head()

X = df.values

# print("Original data sample before encoding (values):", original_df[:5])

columns = df.columns

print("Original X shape: ", original_X.shape)
print("Input dim. of original data: %d" % int(original_X.shape[1]))
print("X (encoded) shape: ", X.shape)
print("Input dim. of encoded data: %d" % int(X.shape[1]))
print()

# order based on feature importance

# Feature importances from Tree-based feature selection

#         1. feature 'feat_11' (100.00000)
#         2. feature 'feat_14' (97.75097)
#         3. feature 'feat_60' (91.44272)
#         4. feature 'feat_34' (87.17911)
#
#         5. feature 'feat_40' (69.08740)

X_indexes = [11, 14, 60, 34]
feat_str = ("'feat_11': {:.0f}, 'feat_14': {:.0f}, 'fet_60': {:.0f}, "
            "'feat_34' {:.0f}")
labels = ['Class_' + str(n) for n in np.arange(1, 10)]

pick_indexes = sample(list(np.arange(0, len(X))), 10)
print("Indexes to randomly pick new data for predictions:", pick_indexes)
print()

clfs = []

print("=== various classifiers")

try:
    clfs.append(
        jl.load('models/OttoG_GBoostingClf_2nd_light_opt_0028.pkl'))
    # this is going to take a while...
    # clfs.append(
    #     jl.load('path/to/OttoG_WhateverClf_2nd_model.pkl'))
except OSError as oe:
    print(oe)
except Exception as e:
    raise e
else:
    print()

    pr.predictions_with_full_estimators(
        clfs, original_X, X, pick_indexes, X_indexes, labels, feat_str)
