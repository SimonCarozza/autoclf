"""Evaluate and train Keras Classifiers."""

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from pandas import read_csv, DataFrame
# from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.wrappers.scikit_learn import KerasClassifier
sys.stderr = stderr

from autoclf.classification import eval_utils as eu
from autoclf import auto_utils as au
from autoclf.classification import neuralnets as nn
from autoclf.classification import train_calibrate as tc
import autoclf.getargs as ga
from autoclf.classification.param_grids_distros import Keras_param_grid
from autoclf.classification.evaluate import create_keras_classifiers
from autoclf.classification.evaluate import create_best_keras_clf_architecture
from pkg_resources import resource_string
from io import StringIO
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits

def select_cv_method():

    is_valid = 0
    choice = 0

    while not is_valid:
        try:
            choice = int(input("Select cv method: [1] Classical CV, [2] Nested-CV?\n"))
            if choice in (1, 2):
                is_valid = 1
            else:
                print("Invalid number. Try again...")
        except ValueError as e:
            print("'%s' is not a valid integer." % e.args[0].split(": ")[1])

    return choice

def select_dataset():

    df = None

    is_valid = 0
    choice = 0

    while not is_valid:
        try:
            choice = int(input("Select dataset: "
                               "[1] German credit, [2] Credit Card, [3] Iris, [4] Digits\n"))
            if choice in (1, 2, 3, 4):
                is_valid = 1

                if choice == 1:
                    names = [
                        'checkin_acc', 'duration', 'credit_history', 'purpose', 'amount',
                        'saving_acc', 'present_emp_since', 'inst_rate', 'personal_status',
                        'other_debtors', 'residing_since', 'property', 'age',
                        'inst_plans', 'housing', 'num_credits', 'job', 'dependents',
                        'telephone', 'foreign_worker', 'status']

                    german_bytes = resource_string(
                        "autoclf", os.path.join("datasets", 'german-credit.csv'))
                    german_file = StringIO(str(german_bytes,'utf-8'))

                    df = read_csv(german_file, header=None, delimiter=" ",
                    names=names)

                    target = 'status'

                    d_name = "GermanCr"
                    
                elif choice == 2:
                    # imb_credit_train.csv
                    data_bytes = resource_string(
                        "autoclf", os.path.join("datasets", 'creditcard.csv'))
                    data_file = StringIO(str(data_bytes,'utf-8'))

                    df = read_csv(data_file, delimiter=",")

                    target = 'Class'

                    d_name = 'ImbCredit'
                elif choice == 3:
                    iris = load_iris()

                    f_names = [name.strip(' (cm)') for name in iris.feature_names]

                    df = DataFrame(data=iris.data, columns=f_names)

                    target = 'class'

                    df[target] = DataFrame(data=iris.target, columns=[target])

                    d_name = 'Iris'
                else:
                    digits = load_digits()

                    df = DataFrame(data=digits.data)

                    target = 'class'

                    df[target] = DataFrame(data=digits.target, columns=[target])

                    d_name = 'Digits'
            else:
                print("Invalid number. Try again...")
        except ValueError as e:
            print("'%s' is not a valid integer." % e.args[0].split(": ")[1])

    return df, target, d_name


# starting program
if __name__ == '__main__':

    plt.style.use('ggplot')

    print("### Probability Calibration Experiment -- CalibratedClassifierCV "
          "with cv=cv (no prefit) ###")
    print()

    d_name = ga.get_name()

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # load data

    data = select_dataset()

    df = data[0]
    target = data[1]

    if d_name is None:
        d_name = data[2]

    print()

    # data exploration

    print("shape: ", df.shape)
    print("df.head():\n", df.head())

    # statistical summary
    description = df.describe()
    print("description - no encoding:\n", description)
    print()

    # input("Enter key to continue... \n")

    # Feature-Feature Relationships
    # scatter_matrix(df)

    print()
    # input("Enter key to continue... \n")

    # feature preprocessing

    sltt = eu.scoring_and_tt_split(df, target, 0.2, seed)

    X_train, X_test, y_train, y_test = sltt['arrays']

    scoring = sltt['scoring']
    Y_type = sltt['target_type']
    labels = sltt['labels']

    print("scoring:", scoring)
    print()
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

    print("y_train:", y_train[:3])
    # input("Enter key to continue... \n")

    print()

    auto_feat_eng_data = eu.auto_X_encoding(sltt, seed)

    print()

    encoding = auto_feat_eng_data['encoding']
    scaler_tuple = auto_feat_eng_data['scaler']
    featselector = auto_feat_eng_data['feat_selector']
    steps = auto_feat_eng_data['steps']
    X_train_transformed, y_train, X_test_transformed, y_test = auto_feat_eng_data['data_arrays']
    X, y = auto_feat_eng_data['Xy']
    train_index, test_index = auto_feat_eng_data['tt_index']

    n_splits = au.select_nr_of_splits_for_kfold_cv()

    print()

    # This cross-validation object is a variation of KFold that returns stratified folds. 
    # The folds are made by preserving the percentage of samples for each class.

    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


    ### reproducing the whole autoclf workflow

    names = []
    results = []

    print("Metric:", scoring)
    print("Calibration of untrained models -- CCCV 2nd")
    print()

    # Evaluation of best modelwith nested CV -- inner: RSCV

    # dict of models and their associated parameters
    # if it comes out that the best model is LogReg, no comparison is needed


    # scoring == 'roc_auc' ==>
    best_atts = eu.best_model_initial_attributes(scoring, n_splits)

    best_score, best_score_dev, best_cv_results, best_model_name = best_atts

    best_exec_time = 31536000    # one year in seconds
    best_model = (best_model_name, None, None)

    Dummy_scores = []

    models_data = []
    names = []
    results = []

    scores_of_best_model = (
        best_score, best_score_dev, best_cv_results,
        best_exec_time, best_model)

    # Start evaluation process

    print()
    print("=== [task] Evaluation of DummyClassifier")
    print()

    wtr = eu.calculate_sample_weight(y_train)

    strategy = 'stratified' # 'most_frequent'

    average_scores_and_best_scores = eu.single_classic_cv_evaluation(
        X_train_transformed, y_train, 'DummyClf_2nd',
        DummyClassifier(strategy=strategy), wtr, scoring, outer_cv,
        dict(), scores_of_best_model, results, names, seed)

    scores_of_best_model = average_scores_and_best_scores[1]

    Dummy_scores.append(scores_of_best_model[0])    # Dummy score -- ROC_AUC
    Dummy_scores.append(scores_of_best_model[1])    # Dummy score std
    Dummy_scores.append(scores_of_best_model[2])    # Dummy cv results
    Dummy_scores.append(scores_of_best_model[3])    # Dummy execution time
    # Dummy model's name and estimator
    Dummy_scores.append(scores_of_best_model[4])

    names = []
    results = []

    print()

    complex_models_and_parameters = dict()
    average_scores_across_outer_folds_complex = dict()

    all_models_and_parameters = dict()

    # Let's add some simple neural network

    print("=== [task] Comparing DummyClassifier to best Keras Clf (NN)")
    print()

    # This is an experiment to check 
    # how different Keras architectures perform
    # to avoid hard-coding NNs, you should determine at least 
    # nr of layers and nr of nodes by using Grid or Randomized Search CV

    input_dim = int(X_train_transformed.shape[1])
    output_dim = 1

    nb_epoch = au.select_nr_of_iterations('nn')

    # evaluate Keras clfs
    
    cv_method = select_cv_method()

    if cv_method == 1:

        batch_size = 32

        complex_models_and_parameters = create_keras_classifiers(
            Y_type, input_dim, output_dim, labels, nb_epoch, batch_size)

        average_scores_and_best_scores = eu.classic_cv_model_evaluation(
            X_train_transformed, y_train, complex_models_and_parameters, scoring,
            outer_cv, average_scores_across_outer_folds_complex,
            scores_of_best_model, results, names, seed)

    else: 

        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        n_iter = au.select_nr_of_iterations()

        keras_clf_name = "KerasClf_2nd"

        keras_nn_model, keras_param_grid = create_best_keras_clf_architecture(
            keras_clf_name, Y_type, labels, input_dim, output_dim, 
            nb_epoch, Keras_param_grid)

        complex_models_and_parameters[keras_clf_name] = (
            keras_nn_model, keras_param_grid)

        average_scores_and_best_scores = eu.nested_rscv_model_evaluation(
            X_train_transformed, y_train, complex_models_and_parameters,
            scoring, n_iter, inner_cv, outer_cv,
            average_scores_across_outer_folds_complex, scores_of_best_model,
            results, names, seed)

    print()
    au.box_plots_of_models_performance(results, names)

    cv_method_name = "Classic" if cv_method == 1 else "Nested"

    print()
    print("=== After %s CV evaluation of Keras NNs..." % cv_method_name)
    print()

    scores_of_best_model = average_scores_and_best_scores[1]

    best_model_name = scores_of_best_model[4][0]
    best_model_estim = scores_of_best_model[4][1]

    best_score = scores_of_best_model[0]
    best_score_dev = scores_of_best_model[1]
    best_cv_results = scores_of_best_model[2]
    # best_brier_score = scores_of_best_model[2]
    best_exec_time = scores_of_best_model[3]

    Dummy_score = Dummy_scores[0]
    Dummy_score_dev = Dummy_scores[1]
    Dummy_cv_results = Dummy_scores[2]
    # Dummy_brier_score = Dummy_scores[3]
    Dummy_exec_time = Dummy_scores[3]

    print()
    print("Currently, best model is '%s' with score '%s': %1.3f (%1.3f)... :" %
          (best_model_name, scoring.strip('neg_'), best_score, best_score_dev))
    if best_model_name in (
            'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
            'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'larger_deep_nn_Clf_2nd', 
            'deeper_nn_Clf_2nd', 'KerasClf_2nd'):
        best_nn_build_fn = scores_of_best_model[4][2]
        print("Best build function:", best_nn_build_fn)
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

        if labels is not None:
            print("You have labels:", labels)
            all_models_and_parameters['labels'] = labels

        print("Defined dictionary with models, parameters and related data.")
        print()

        if cv_method == 1:

            tc.calibrate_best_model(
                X, y, X_train_transformed, X_test_transformed,
                y_train, y_test, auto_feat_eng_data['tt_index'], 
                preprocessing, scores_of_best_model,
                all_models_and_parameters, n_splits, nb_epoch,
                scoring, models_data, d_name, seed)
        else:

            tc.tune_calibrate_best_model(
                X, y, X_train_transformed, X_test_transformed,
                y_train, y_test, auto_feat_eng_data['tt_index'], 
                preprocessing, scores_of_best_model,
                all_models_and_parameters, n_splits, n_iter, nb_epoch,
                scoring, models_data, d_name, seed)

    else:
        sys.exit("Your best classifier is not a good classifier.")

    input("=== [End Of Program] Enter key to continue... \n")