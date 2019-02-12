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
from sklearn.datasets import load_digits
import pathlib as pl

# Let's make predictions and print their associated probabilities

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load data

digits = load_digits()

df = DataFrame(data=digits.data)

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

print("Original data sample before encoding (values):", original_X[:5])

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

#         1. feature '28' (100.00000)
#         2. feature '36' (99.99035)
#         3. feature '42' (98.93417)
#         4. feature '26' (97.75111)
#         5. feature '21' (87.22771)
#         6. feature '43' (81.67973)
#   
#         7. feature '33' (65.42270)

X_indexes = [28, 36, 42, 26, 21, 43]
feat_str = ("'28': {:.3f}, '36': {:.3f}, '42': {:.3f}, '26' {:.3f}, "
            "'21' {:.3f}, '43' {:.3f}")

labels = list(np.arange(10))

pick_indexes = sample(list(np.arange(0, len(X))), 10)
print("Indexes to randomly pick new data for predictions:", pick_indexes)
print()

clfs = []

print("=== various classifiers")

try:
    clfs.append(
        jl.load('models/Digits_LogRClf_2nd_final_calib_rscv_0029.pkl'))
    clfs.append(
        jl.load('models/Digits_QDAClf_2nd_final_calib_rscv_0937.pkl'))
    # KerasEstimator = jl.load(
    #     'path/to/feature_transformer.pkl')
    # KerasEstimator.steps.append(
    #     ('KerasClf', load_model('path/to/keras_clf.h5')))
    KerasEstimator = jl.load(
        'models/Digits_deep_nn_Clf_2nd_feateng_for_keras_model_0428.pkl')
    KerasEstimator.steps.append(
        ('Digits_deep_nn_Clf_2nd', load_model(
            'models/Digits_deep_nn_Clf_2nd_None_0428.h5')))
    KerasEstimator = jl.load(
        'models/Digits_KerasClf_2nd_feateng_for_keras_model_0546.pkl')
    KerasEstimator.steps.append(
        ('KerasClf_2nd_refitted_rscv_0888', load_model(
            'models/Digits_KerasClf_2nd_refitted_rscv_0546.h5')))
    clfs.append(KerasEstimator)
except OSError as oe:
    print(oe)
except Exception as e:
    raise e
else:
    print()

    # Keras models return an array of probabilities,
    # each associated to a given label

    pr.predictions_with_full_estimators(
        clfs, original_X, X, pick_indexes, X_indexes, labels, feat_str)
