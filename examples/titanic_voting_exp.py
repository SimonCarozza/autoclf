from pandas import read_csv
# from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

from autoclf.classification import eval_utils as eu

from pkg_resources import resource_string
from io import StringIO
import autoclf.auto_utils as au
from autoclf.encoding import labelenc as lc
from autoclf.classification import param_grids_distros as pgd

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# starting program
if __name__ == '__main__':


    print("### Probability Calibration Experiment -- CalibratedClassifierCV with cv=cv (no prefit) ###")

    print()
    print("=== Plotting calibration curves.")

    # curdir = os.path.dirname(__file__)

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # load data

    titanic_bytes = resource_string(
            "autoclf", os.path.join("datasets", 'titanic_train.csv'))
    titanic_file = StringIO(str(titanic_bytes,'utf-8'))

    names = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp',
             'Parch','Ticket','Fare','Cabin','Embarked']

    df = read_csv(
        titanic_file, delimiter=",",
        # header=0, names=names,
        na_values={'Age': '', 'Cabin': '', 'Embarked': ''},
        dtype={'Name': 'category', 'Sex': 'category',
               'Ticket': 'category', 'Cabin': 'category',
               'Embarked': 'category'})

    # data exploration

    print("shape: ", df.shape)
    print()
    print("df.head():\n", df.head())
    print()

    # statistical summary
    description = df.describe()
    print("description - no encoding:\n", description)
    print()

    # set_option("display.mpl_style", "default")
    plt.style.use('ggplot')

    # Feature-Feature Relationships
    # scatter_matrix(df)

    print()

    # too many missing values in 'Cabin' columns: about 3/4
    print("Dropping 'Cabin' column -- too many missing values")
    df.drop(['Cabin'], axis=1, inplace=True)

    target = 'Survived'

    # feature engineering

    sltt = eu.scoring_and_tt_split(df, target, 0.2, seed, Xy=True)

    scoring = sltt['scoring']
    Y_type = sltt['target_type']
    labels = sltt['labels']
    X_train, X_test, y_train, y_test = sltt['arrays']
    train_idx, test_idx = sltt['tt_index']

    print("Classes:", labels)

    print()
    print("X_train shape: ", X_train.shape)
    print("X_train -- first row:", X_train.values[0])
    print("y_train shape: ", y_train.shape)
    print()

    print("X_test shape: ", X_test.shape)
    print("X_test -- first row:", X_test.values[0])
    print("y_test shape: ", y_test.shape)
    print()

    print(y_train[:3])
    print("Y_train unique values:", len(np.unique(y_train)))
    # input("Enter key to continue... \n")

    print()
    print("scoring:", scoring)
    print()

    # auto_feat_eng_data = auto_X_encoding_only(sltt['arrays'], seed)
    # no feature selection: f_sel=False
    auto_feat_eng_data = eu.auto_X_encoding(sltt, seed)

    print()

    encoding = auto_feat_eng_data['encoding']
    scaler_tuple = auto_feat_eng_data['scaler']
    # uncomment if using cf.auto_X_encoding()
    try:
       featselector = auto_feat_eng_data['feat_selector']
    except KeyError as ke:
        print("No transformer from feat. selection here, "
              "this must be plain encoding.")
    except Exception as e:
        print(e)
    
    steps = auto_feat_eng_data['steps']

    X_train_transformed, y_train, X_test_transformed, y_test = \
        auto_feat_eng_data['data_arrays']
    X, y = auto_feat_eng_data['Xy']

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=seed)

    train_idx, val_idx = (None, None)

    for trn, val in sss.split(X_train_transformed, y_train):
        train_idx, val_idx = trn, val

    X_train_transformed, X_valid \
    = X_train_transformed[train_idx], X_train_transformed[val_idx]
    y_train, y_valid = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # X_train_transformed, X_valid, y_train, y_valid = train_test_split(
    #     X_train_transformed, y_train, stratify=y_train, test_size=0.2, 
    #     random_state=seed)

    n_splits = au.select_nr_of_splits_for_kfold_cv()
    n_iter = au.select_nr_of_iterations()

    print()

    # This is a variation of KFold that returns stratified folds.
    # The folds are made by preserving the percentage of samples for each class.
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # reproducing the whole workflow

    names = []
    results = []

    # print("y_train sample:", y_train)
    print("Y_train unique values:", len(np.unique(y_train)))

    wtr = eu.calculate_sample_weight(y_train)

    models_and_parameters = pgd.full_search_models_and_parameters

    # take 3 models w attribute predict_proba() and
    # create VotingClf3 or just select suitable Clfs
    # these result from a previous nested cv run

    top_3_models_data = {
        'GBoostingClf_2nd': (
            None, None, None, models_and_parameters['GBoostingClf_2nd'][0],
            models_and_parameters['GBoostingClf_2nd'][1]),
        'AdaBClf_2nd': (
            None, None, None, models_and_parameters['AdaBClf_2nd'][0],
            models_and_parameters['AdaBClf_2nd'][1]),
        'LDAClf_2nd': (
            None, None, None, models_and_parameters['LDAClf_2nd'][0],
            models_and_parameters['LDAClf_2nd'][1])
    }

    estimators = [k for k, v in top_3_models_data.items()]
    for i, est in enumerate(estimators):
        print("Estimator nr. %d: %s" % (i, est))

    print()

    # I already know lr_pred_score's value for this Titanic problem configuration
    lr_pred_score = 0.185

    vc3_estimators = eu.calibrate_estimators_for_soft_voting(
        top_3_models_data, X_train_transformed, y_train, X_valid, y_valid,
        X_test_transformed, y_test, n_splits, n_iter, scoring, 
        lr_pred_score, outer_cv, tuning_method=''
        )

    vclf3 = VotingClassifier(vc3_estimators, voting='soft')

    vc3_estimator_name = "VClf_3_2nd"

    steps = []

    # steps.append(scaler_tuple)
    # if you're using feature selector, you might not need cf.auto_encoding_X()
    # but you should hard-code all the encoding + feature importance plotting
    # steps.append(featselector)
    steps.append((vc3_estimator_name, vclf3))

    vclf3_pipeline = Pipeline(steps)

    temp_pipeline = vclf3_pipeline

    # vclf3_pipeline <-- vclf3

    # ...then tune VotingClassifier's weights

    vc3_params = {
        vc3_estimator_name + '__'
        + k: v for k, v in pgd.VC_3_param_grid.items()}

    vc3_n_iter = au.check_search_space_of_params(n_iter, vc3_params)

    best_vclf3_params = eu.tune_and_evaluate(
        temp_pipeline, X_train_transformed, y_train, X_test_transformed,
        y_test, n_splits, vc3_params, vc3_n_iter, scoring, [], refit=False,
        random_state=seed
        )

    temp_pipeline.set_params(**best_vclf3_params)

    tuned_vclf3_pipeline = temp_pipeline

    tuned_vclf3_pipeline.fit(X_train_transformed, y_train)

    predicted = tuned_vclf3_pipeline.predict(X_test_transformed)

    print()

    if Y_type == 'binary':
        tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
        print()
        print("Test errors for '%s'" % vc3_estimator_name)
        print("\ttrue negatives: %d" % tn)
        print("\tfalse positives: %d" % fp)
        print("\tfalse negatives: %d" % fn)
        print("\ttrue positives: %d" % tp)
        print()
        print("Classification report for '%s'\n" %
              vc3_estimator_name, classification_report(y_test, predicted))

    print()
    print("X_train_transformed shape: ", X_train_transformed.shape)
    print("X_test_transformed shape: ", X_test_transformed.shape)
    print()

    # doing the following to avoid inverse transform X_train, X_test

    final_idx = np.concatenate((train_idx, test_idx), axis=0)
    X_final = X.iloc[final_idx]
    y_final = y.iloc[final_idx]

    print("X shape: ", X_final.shape)
    print("y shape: ", y_final.shape)
    print("X sample:\n", X_final[:3])
    print("Y sample:\n", y_final[:3])
    print()

    # finalize Votingclassifier and save
