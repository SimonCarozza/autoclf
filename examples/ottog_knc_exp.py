from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

from pandas import read_csv, DataFrame
import numpy as np

import os
import sys
from autoclf import auto_utils as au
from autoclf.classification import eval_utils as eu
from autoclf.classification import param_grids_distros as pgd
from autoclf.classification import train_calibrate as tc
import autoclf.getargs as ga
from pkg_resources import resource_string
from io import StringIO


# starting program
if __name__ == '__main__':

    print()
    print("### Probability Calibration Experiment -- CalibratedClassifierCV "
          "with cv=cv (no prefit) ###")
    print()

    d_name = ga.get_name()

    if d_name is None:
        d_name = "OttoG"

    seed = 7
    np.random.seed(seed)

    try:
        df = read_csv("datasets/otto_group_train.csv", delimiter=",")
    except FileNotFoundError as fe:
        ottog_bytes = resource_string(
            "autoclf", os.path.join("datasets", 'otto_group_train.csv'))
        ottog_file = StringIO(str(ottog_bytes,'utf-8'))

        df = read_csv(ottog_file, delimiter=",")
    except Exception as e:
        raise e

    print(df.shape)

    print("Dataframe description - no encoding:\n", df.describe())
    print()

    print()
    print("=== [task] Train-test split + early pre-processing.")
    print()

    # Missing Attribute Values: None

    ###
    df = df.drop(['id'], axis=1)
    print(df.shape)

    description = df.describe()
    print("Description - no encoding:\n", description)

    print()

    target = 'target'

    # feature engineering

    sltt = eu.scoring_and_tt_split(df, target, 0.2, seed)

    X_train, X_test, y_train, y_test = sltt['arrays']
    scoring = sltt['scoring']
    Y_type = sltt['target_type']
    labels = sltt['labels']

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
    # input("Enter key to continue... \n")

    print()
    print("scoring:", scoring)
    print()

    auto_feat_eng_data = eu.auto_X_encoding(sltt, seed)

    encoding = auto_feat_eng_data['encoding']
    scaler_tuple = auto_feat_eng_data['scaler']
    featselector = auto_feat_eng_data['feat_selector']
    steps = auto_feat_eng_data['steps']
    X_train_transformed, y_train, X_test_transformed, y_test =\
    auto_feat_eng_data['data_arrays']
    X, y = auto_feat_eng_data['Xy']
    train_index, test_index = auto_feat_eng_data['tt_index']

    n_splits = au.select_nr_of_splits_for_kfold_cv()
    n_iter = au.select_nr_of_iterations()

    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    ### reproducing the whole autoclf workflow

    names = []
    results = []

    print("Metric:", scoring)
    print("Calibration of untrained models -- CCCV 2nd")
    print()

    best_atts = eu.best_model_initial_attributes(scoring, n_splits)

    best_score, best_score_dev, best_cv_results, best_model_name = best_atts

    best_exec_time = 31536000    # one year in seconds
    best_model = (best_model_name, None, None)

    Dummy_scores = []

    models_data = []
    names = []
    results = []

    scores_of_worst_model = (best_score, best_score_dev, best_cv_results,
                            best_exec_time, best_model)

    scores_of_best_model = scores_of_worst_model

    average_scores_and_best_scores = dict()
    average_scores_and_best_scores[best_model_name] \
    = (best_score, best_score_dev, best_exec_time, best_model, {})

    # Start evaluation process

    print()
    print("=== [task] Evaluation of DummyClassifier")
    print()

    wtr = eu.calculate_sample_weight(y_train)

    strategy = 'stratified'  # 'most_frequent'

    evaluation_result = eu.single_classic_cv_evaluation(
        X_train_transformed, y_train, 'DummyClf_2nd',
        DummyClassifier(strategy=strategy), wtr, scoring, outer_cv,
        dict(), scores_of_best_model, results, names, seed)

    average_scores_and_best_scores = evaluation_result[0]
    scores_of_best_model = evaluation_result[1]

    Dummy_scores.append(scores_of_best_model[0])    # Dummy score -- ROC_AUC
    Dummy_scores.append(scores_of_best_model[1])    # Dummy score std
    Dummy_scores.append(scores_of_best_model[2])    # Dummy cv results
    Dummy_scores.append(scores_of_best_model[3])    # Dummy execution time
    # Dummy model's name and estimator
    Dummy_scores.append(scores_of_best_model[4])

    names = []
    results = []

    print()

    all_models_and_parameters = dict()

    # replace with name from pgd.full_search_models_and_parameters
    test_model_name = 'model_from_param_grids_distros'  # 'KNeighborsClf_2nd'

    print("=== [task] Comparing DummyClassifier to KNeighborsClassifier")
    print()

    evaluation_result = eu.single_nested_rscv_evaluation(
        X_train_transformed, y_train, test_model_name,
        pgd.full_search_models_and_parameters[test_model_name][0], 
        pgd.full_search_models_and_parameters[test_model_name][1],
        wtr, scoring, n_iter, inner_cv, outer_cv, 
        average_scores_and_best_scores, scores_of_best_model, 
        results, names, seed)

    print()
    au.box_plots_of_models_performance(results, names)

    print()
    print("=== After Non-nested CV evaluation of %s..." % test_model_name)
    print()

    scores_of_best_model = evaluation_result[1]

    best_model_name = scores_of_best_model[4][0]
    best_model_estim = scores_of_best_model[4][1]

    best_score = scores_of_best_model[0]
    best_score_dev = scores_of_best_model[1]
    best_cv_results = scores_of_best_model[2]
    best_exec_time = scores_of_best_model[3]

    Dummy_score = Dummy_scores[0]
    Dummy_score_dev = Dummy_scores[1]
    Dummy_cv_results = Dummy_scores[2]
    Dummy_exec_time = Dummy_scores[3]

    print()
    print("Currently, best model is '%s' with score '%s': %1.3f (%1.3f)... :" %
          (best_model_name, scoring.strip('neg_'), best_score, best_score_dev))
    print("... execution time: %.2fs" % best_exec_time)
    # print("and prediction confidence: %1.3f" % best_brier_score)
    print()

    if best_model_name != 'DummyClf_2nd':
        # It's assumed best model's performance is
        # satistically better than that of DummyClf on this dataset
        print("DummyClassifier's scores -- '%s': %1.3f (%1.3f)" % (
            scoring.strip('neg_'), Dummy_score, Dummy_score_dev))
        print("'%s' does better than DummyClassifier." % best_model_name)
        if best_exec_time < Dummy_exec_time:
            print("'%s' is quicker than DummyClf." % best_model_name)
        print()
        print()
        input("Press key to continue...")

        preprocessing = (encoding, scaler_tuple, featselector)

        all_models_and_parameters[best_model_name] = (
            best_model, pgd.full_search_models_and_parameters[best_model_name][1])

        if labels is not None:
            print("You have labels:", labels)
            all_models_and_parameters['labels'] = labels

        print("Defined dictionary with models, parameters and related data.")
        print()

        tc.tune_calibrate_best_model(
            X, y, X_train_transformed, X_test_transformed,
            y_train, y_test, auto_feat_eng_data['tt_index'], 
            preprocessing, scores_of_best_model,
            all_models_and_parameters, n_splits, n_iter, 0,
            scoring, models_data, d_name, seed)
    else:
        sys.exit("Your best classifier is not a good classifier.")

    input("=== [End Of Program] Enter key to continue... \n")