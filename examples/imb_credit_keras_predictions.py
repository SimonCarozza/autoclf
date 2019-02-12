from pandas import read_csv
import os

from keras.models import load_model
from sklearn.externals import joblib as jl
import sklearn
import numpy as np
from random import sample

# autoclf classification's module
from autoclf.encoding import labelenc as lc
from autoclf.classification import predict as pr
from autoclf.classification.eval_utils import columns_as_type_float
from pkg_resources import resource_string
from io import StringIO

# Let's make predictions and print their associated probabilities

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load data

imbcr_bytes = resource_string(
    "autoclf", os.path.join("datasets", 'creditcard.csv'))
imbcr_file = StringIO(str(imbcr_bytes,'utf-8'))

# loading just 3% of full dataset
df = read_csv(imbcr_file, delimiter=",").sample(frac=0.03)

df.drop(['Class'], axis=1, inplace=True)

# data exploration

print("shape: ", df.shape)

print()

# replace missing valus
if df.isnull().values.any():
    print("Null values here... replacing them.")
    df.fillna(method='pad', inplace=True)
    df.fillna(method='bfill', inplace=True)

# print("Original data sample before encoding:", df[:5])

df = columns_as_type_float(df)

description = df.describe()
print("Description - no encoding:\n", description)

print()
# input("Press key to continue... \n")

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

pick_indexes = sample(range(0, len(X)), 10)
print("Indexes to randomly pick new data for predictions:", pick_indexes)
print()

# feat. importance order: 

# Feature importances from Tree-based feature selection

#         1. feature 'V14' (100.00000)
#         2. feature 'V16' (97.62559)
#         3. feature 'V12' (71.36555)
#         4. feature 'V17' (70.11398)
#         5. feature 'V10' (65.10145)

X_indexes = [14, 16, 17, 10, 11]
feat_str = ("'V14': {:.2f}, 'V16': {:.2f}, ''V12' {:.2f}, 'V17': {:.2f}, 'V10: {:.2f}")
neg = "good_user"
pos = "bad_user"
bin_event = (neg, pos)

print("======= Transfomer + Keras model from ev.calibrate_best_model()")

clfs = []

try:
    # KerasEstimator = jl.load(
    #     'path/to/feature_transformer.pkl')
    # KerasEstimator.steps.append(
    #     ('KerasClf', load_model('path/to/keras_clf.h5')))
    KerasEstimator = jl.load(
        'models/ImbCredit_deep_nn_Clf_2nd_feateng_for_keras_model_0975.pkl')
    KerasEstimator.steps.append(
        ('KerasClf_2nd_None_0975', load_model(
            'models/ImbCredit_deep_nn_Clf_2nd_None_0975.h5')))
    clfs.append(KerasEstimator)
except FileNotFoundError as fe:
    print(fe)
except Exception:
    raise
else:
    pr.predictions_with_full_estimators(
        clfs, original_X, X, pick_indexes, X_indexes, bin_event, feat_str)

print()
print()

print("======= Transfomer + Keras model from ev.tune_calibrate_best_model()")

try:
    KerasEstimator = jl.load(
        'models/ImbCredit_KerasClf_2nd_feateng_for_keras_model_0286.pkl')
    KerasEstimator.steps.append(
        ('KerasClf_2nd_rscv_0286', load_model(
            'models/ImbCredit_KerasClf_2nd_rscv_0286.h5')))
except OSError as oe:
    print(oe)
except Exception as e:
    raise e
else:

    print("Pipeline of transformer + Keras Clf:")
    print(KerasEstimator)
    print()

    ker_name = 'KerasClf_2nd_rscv_0286'

    predicted = 0
    try:
        pr.prediction_with_single_estimator(
            ker_name, KerasEstimator, original_X, X, 
            pick_indexes, X_indexes, bin_event, feat_str)
    except TypeError as te:
        print(te)
    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(e)
    else:
        predicted = 1
    finally:
        if predicted:
            print("Prediction with '%s' successful." % ker_name)
        else:
            print("Prediction with '%s' failed." % ker_name)
