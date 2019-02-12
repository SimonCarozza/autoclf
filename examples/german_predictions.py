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

names = [
    'checkin_acc', 'duration', 'credit_history', 'purpose', 'amount',
    'saving_acc', 'present_emp_since', 'inst_rate', 'personal_status',
    'other_debtors', 'residing_since', 'property', 'age',
    'inst_plans', 'housing', 'num_credits', 'job', 'dependents',
    'telephone', 'foreign_worker', 'status']

german_bytes = resource_string(
    "autoclf", os.path.join("datasets", 'german-credit.csv'))
german_file = StringIO(str(german_bytes,'utf-8'))

df = read_csv(german_file, header=None, delimiter=" ", names=names)

df.drop(['status'], axis=1, inplace=True)

# data exploration

print("shape: ", df.shape)

print()


# replace missing valus in 'Age', 'Embarked'
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

# feat. importance order: checkin_acc, amount, duration, age

X_indexes = [0, 4, 1, 12]
feat_str = ("'checkin_acc': {}, 'amount': {:.0f}, 'duration': {:.0f}, 'age: {:.0f}")
neg = "good_user"
pos = "bad_user"
bin_event = (neg, pos)

print()
print("======= Tuned ensemble of GaussianNBs from cf.tune_calibrate_best_model()")

try:
    gauss_est = jl.load('models/GermanCr_Bagging_GaussianNBClf_2nd_final_nocalib_rscv_0982.pkl')
except FileNotFoundError as fe:
    print(fe)
except Exception:
    raise
else:
    pr.prediction_with_single_estimator(
            'Bagging_GaussianNBClf', gauss_est, original_X, X, pick_indexes, X_indexes,
            bin_event, feat_str)

print()
print()

clfs = []

print("======= Transfomer + Keras model from cf.tune_calibrate_best_model()")

try:
    # KerasEstimator = jl.load(
    #     'path/to/feature_transformer.pkl')
    # KerasEstimator.steps.append(
    #     ('KerasClf', load_model('path/to/keras_clf.h5')))
    KerasEstimator = jl.load(
        'models/GermanCr_KerasClf_2nd_feateng_for_keras_model_0368.pkl')
    KerasEstimator.steps.append(
        ('KerasClf_2nd_rscv_0368', load_model(
            'models/GermanCr_KerasClf_2nd_rscv_0368.h5')))
    clfs.append(KerasEstimator)
except OSError as oe:
    print(oe)
except Exception as e:
    raise e
else:

    print("Pipeline of transformer + Keras Clf:")
    print(KerasEstimator)
    print()

print("=== refitted Keras classifier from tune_evaluate()")

try:
    # KerasEstimator = jl.load(
    #     'path/to/feature_transformer.pkl')
    # KerasEstimator.steps.append(
    #     ('KerasClf', load_model('path/to/keras_clf.h5')))
    KerasEstimator = jl.load(
        'models/GermanCr_KerasClf_2nd_feateng_for_keras_model_0888.pkl')
    KerasEstimator.steps.append(
        ('KerasClf_2nd_refitted_rscv_0888', load_model(
            'models/GermanCr_KerasClf_2nd_refitted_rscv_0888.h5')))
    clfs.append(KerasEstimator)
except OSError as oe:
    print(oe)
except Exception as e:
    raise e
else:
    print("Pipeline of transformer + Keras Clf:")
    print(KerasEstimator)
    print()

print()

pr.predictions_with_full_estimators(
    clfs, original_X, X, pick_indexes, X_indexes, bin_event, feat_str)