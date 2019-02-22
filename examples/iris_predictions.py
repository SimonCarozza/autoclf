from pandas import read_csv
import os
import sys
from autoclf.encoding import labelenc as lc
from keras.models import load_model
from sklearn.externals import joblib as jl
import sklearn
import numpy as np

# autoclf classification's module
from autoclf.classification import predict as pr
from pkg_resources import resource_string
from io import StringIO
import pathlib as pl

# Let's make predictions and print their associated probabilities

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# load data

names = ['seplength', 'sepwidth', 'pellength', 'pellwidth', 'class']

# moddir = os.path.dirname(__file__)
# iris_file = os.path.join(moddir, "datasets", "iris.csv")
iris_bytes = resource_string(
    "autoclf", os.path.join("datasets", 'iris.csv'))
iris_file = StringIO(str(iris_bytes,'utf-8'))

# df = read_csv('autoclf\\datasets\\iris.csv', delimiter=",", names=names)
df = read_csv(iris_file, delimiter=",", names=names)

print("shape: ", df.shape)

print()

df.drop(['class'], axis=1, inplace=True)

"""
description = df.describe()
print("Once again, description - no encoding:\n", description)
"""

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
X_indexes = [3, 2, 0, 1]
feat_str = ("'pellwidth': {:.1f}, 'pellength': {:.1f}, 'seplength': {:.1f}, "
            "sepwidth '{:.1f}'")

labels = ('setosa', 'versicolor', 'virginica')

# indexes = sample(range(0, len(X)), 10)
pick_indexes = list(np.arange(0, 150, 15))
print("Indexes to randomly pick new data for predictions:", pick_indexes)
print()

clfs = []

try:
    if sklearn.__version__[:4] == '0.19':
        # Using Python 3's pathlib module
        moddir_36 = pl.Path(__file__).parent.resolve()
        rf_pkl = os.path.join(
            moddir_36, "models",
            "iris_RandomForestClf_2nd_final_nocalib_rscv_0039.pkl")
        clfs.append(jl.load(rf_pkl))
    elif sklearn.__version__[:4] == '0.20':
        moddir_36 = pl.Path(__file__).parent.resolve()
        lr_pkl = os.path.join(
            moddir_36, "models",
            "iris_LogRClf_2nd_final_calib_rscv_0429.pkl")
        clfs.append(jl.load(lr_pkl))

        kr_estim = jl.load(
            "models/Iris_KerasClf_2nd_feateng_for_keras_model_0289.pkl")
        kr_estim.steps.append(
            ('Iris_KerasClf_2nd_0289', 
            load_model("models/Iris_KerasClf_2nd_refitted_rscv_0289.h5")))
        clfs.append(kr_estim)
except FileNotFoundError as fne:
    print(fne)
except Exception:
    raise
else:
    pr.predictions_with_full_estimators(
        clfs, original_X, X, pick_indexes, X_indexes, labels, feat_str)
