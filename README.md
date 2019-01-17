# autoclf

**autoclf** is a machine learning mini-framework for **automatic tabular data classification** with calibrated estimators, and exploration of the scikit-learn and Keras ecosystems.

autoclf is a humble bunch of small experiments glued together with the ambitious aim of **full automation**, and as such, it's a real mini-framework featuring six modules to preprocess data, train, calibrate, save models and make predictions.

autoclf keeps an eye on **feature engineering**, so methods have been developed to train-test-split data, automatically establish a classification metric - 'roc_auc' vs 'log_loss_score' - based on binary or multiclass target type and get class labels for you to inject your custom hacks without the risk of data leakage from test to train set before Label/One-Hot encoding them separately.


## Installation

* Install from Anaconda cloud

Install **python 3.6** first, then install scikit-learn, pandas, matplotlib, keras. If you have [Anaconda Python](https://www.anaconda.com/download/) installed, which I recommend to ease installations in Windows, you can also install py-xgboost, a sklearn wrapper for the popular distributed gradient boosting library [XGBoost](https://xgboost.readthedocs.io/en/latest/index.html), which is optional to autoclf's working.

`conda create env -n clf python=3.6`

`activate clf`

`conda install scikit-learn matplotlib pandas keras py-xgboost`

`conda install autoclf`

* Install from Github

First, clone autoclf using git:

`git clone https://github.com/SimonCarozza/autoclf.git`

Then, cd to the autoclf folder and run the install command:

`cd autoclf`
`python setup.py install`

You can go and find pickled models in folder "examples" to make predictions with both sklearn 0.19 and 0.20 depending on the version you installe on your PC.


## How to use autoclf

autoclf classifies **small and medium datasets** -- from 100 up to 1,000,000 samples -- and makes use of sklearn's jobs to parallelize calculations.

1. Load data as a pandas dataframe, 

   ```python
   import pandas as pd

   df = pd.read_csv(
       'datasets/titanic_train.csv', delimiter=",",
       na_values={'Age': '', 'Cabin': '', 'Embarked': ''},
       dtype={'Name': 'category', 'Sex': 'category',
       'Ticket': 'category', 'Cabin': 'category',
       'Embarked': 'category'})
    ```

1. split data into the usual train-test subsets:

   ```python
   from autoclf.classification import eval_utils as eu

   target = 'Survived'

   sltt = eu.scoring_and_tt_split(df, target, test_size=0.2, random_state=seed)

   X_train, X_test, y_train, y_test = sltt['arrays']
   ```

   1. get scoring, target type and class labels,

   ```python
   scoring = sltt['scoring']
   Y_type = sltt['target_type']
   labels = sltt['labels']
   ```

   1. automatically reduce dataframe to a digestible size (optional)

   ```python
   learnm, ldf = eu.learning_mode(df)
   odf = None

   if len(ldf.index) < len(df.index):

       odf = df
       df = ldf
   ```

1. do your custom feature engineering,

1. automatically (Label / One-Hot) encode subsets and include them into a single dict, 

   ```python
   auto_feat_eng_data = eu.auto_X_encoding(sltt, seed)
   ```

1. pass engineered dict 'sltt' to evaluation method to run a **nested or a classical cross-validation** of estimators with eventual probability calibration.

   ```python
   from autoclf.classification import evaluate as eva
   eva.select_evaluation_strategy(
       auto_feat_eng_data, scoring, Y_type, labels=labels,
       d_name='titanic', random_state=seed, learn=learnm)
   ```

1. make predictions as usual using sklearn's or Keras' predict() method or play with autoclf's predict module.

   ```python
   from pandas import read_csv
   from keras.models import load_model
   from sklearn.externals import joblib as jl
   import numpy as np
   from random import sample

   # auto_learn classfication's module
   from autoclf.encoding import labelenc as lc
   from autoclf.classification import predict as pr

   # ... load test data with read_csv

   original_df = df

   original_X = original_df.values

   # encode the dataframe
   df = lc.dummy_encode(df)

   X = df.values

   pick_indexes = sample(range(0, len(X)), 10)

   # feat. importance order: sex, age, ticket, fare, name
   # Name: %s, sex '%s', age '%d', fare '%.1f'
   X_indexes = [2, 3, 4, 7, 8]
   feat_str = ("'Name': {}, 'sex': {}, 'age': {:.0f}, 'ticket': {}, 'fare': {:.1f}")
   neg = "dead"
   pos = "survived"
   bin_event = (neg, pos)

   clfs.append(
       jl.load('models/titan_AdaBClf.pkl'))
   clfs.append(
       jl.load('models/titan_Bagging_SVMClf.pkl'))

   pr.predictions_with_full_estimators(
       clfs, original_X, X, pick_indexes, X_indexes, bin_event, feat_str)
   ```

autoclf works with **both sklearn=0.19 and sklearn=0.20**, but if you switch versions to do different things with resulting models, they could break. In short, if you train and save a model using sklearn=0.19, don't switch to sklearn=0.20 to make predictions.


## Disclaimer

autoclf is born out of the simple need of **trying out sklearn's and Keras' features**, and as such, it's full of hacks and uses processes that could be replaced with faster solutions. 

As a toy concept mini-framework, autoclf **has not been tested following a [TDD](https://en.wikipedia.org/wiki/Test-driven_development) approach**, so it's not guaranteed to be stable and is **not aimed to nor ready for production**.

autoclf has been developed for **Windows 10** but has proved to work smoothly in Ubuntu Linux as well.
