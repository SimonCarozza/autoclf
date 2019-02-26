"""
This module contains functions to perform non-nested and nested cv.

This module contains a function to select a strategy for models evaluation.

"""

from sklearn.ensemble import BaggingClassifier
from . import param_grids_distros as pgd
from . import neuralnets as nn
from operator import itemgetter

import numpy as np
from scipy.stats import randint as sp_randint
from .. import auto_utils as au
from . import train_calibrate as tc

from . import eval_utils as eu
import autoclf.getargs as ga
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.wrappers.scikit_learn import KerasClassifier
sys.stderr = stderr
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")


def create_keras_classifiers(
    y_type, input_dim, labels, nb_epoch, batch_size):

    keras_clf_fcts = dict(
            baseline_nn_default_Clf_2nd=(
                nn.baseline_nn_model_multiclass, nn.baseline_nn_model),
            baseline_nn_smaller_Clf_2nd=(
                nn.baseline_nn_smaller_model_multiclass, nn.baseline_nn_smaller_model),
            larger_nn_Clf_2nd=(nn.larger_nn_model_multiclass, nn.larger_nn_model),
            deep_nn_Clf_2nd=(nn.deep_nn_model_multiclass, nn.deep_nn_model),
            deeper_nn_Clf_2nd=(nn.deeper_nn_model_multiclass, nn.deeper_nn_model)
            )

    if input_dim < 15:
            keras_clf_fcts['larger_deep_nn_Clf_2nd'] = (
                nn.larger_deep_nn_model_multiclass, nn.larger_deep_nn_model)

    names_and_models=dict()

    output_dim = 1

    if y_type == 'multiclass':

        if labels is None:
            raise ValueError("%r is not a valid type for var 'labels'" % labels)
        elif not isinstance(labels, list):
            raise TypeError("Multiclass keras models need a list of string labels.")
        else:
            output_dim = len(labels)

        for k, v in keras_clf_fcts.items():
            names_and_models[k] = KerasClassifier(
            build_fn=v[0], nb_epoch=nb_epoch,
            input_dim=input_dim, output_dim=output_dim, batch_size=batch_size,
            verbose=0)

    else:

        for k, v in keras_clf_fcts.items():
            names_and_models[k] = KerasClassifier(
            build_fn=v[1], nb_epoch=nb_epoch,
            input_dim=input_dim, batch_size=batch_size,
            verbose=0)

    return names_and_models


def create_best_keras_clf_architecture(
        keras_clf_name, y_type, labels, input_dim, nb_epoch, keras_param_grid):
    """
    Find KerasClf best architecture using

    ------
    """

    output_dim = 1

    for n in np.arange(0, 3):
        keras_param_grid[keras_clf_name + '__units_' + str(n)] = sp_randint(
            input_dim, 5*input_dim)

    if y_type == 'multiclass':

        if labels is None:
            raise ValueError("%r is not a valid type for var 'labels'" % labels)
        elif not isinstance(labels, list):
            raise TypeError("Multiclass keras models need a list of string labels.")
        else:
            output_dim = len(labels)

        keras_nn_model = KerasClassifier(
            build_fn=nn.tunable_deep_nn_multiclass, nb_epoch=nb_epoch,
            input_dim=input_dim, output_dim=output_dim,
            verbose=0)

    else:

        # you need KerasClassifier wrapper to use Keras models in sklearn

        keras_nn_model = KerasClassifier(
            build_fn=nn.tunable_deep_nn, nb_epoch=nb_epoch,
            input_dim=input_dim, verbose=0)

    return (keras_nn_model, keras_param_grid)


def models_and_params_for_classic_cv_tasks(y_type, scoring, labels):
    models_and_params = dict()

    if y_type == 'binary':
        models_and_params = pgd.starting_point_models_and_params
        # models_and_params['LogRClf_2nd'].set_params(
        #     solver='liblinear')

    else:
        # if y_type == 'multiclass':

        if scoring == 'neg_log_loss':               
            for k, v in pgd.starting_point_models_and_params.items():
                if hasattr(v, 'predict_proba'):
                    print(k)
                    models_and_params[k] = v

            # solver='saga', penalty='l1'
            # models_and_params['LogRClf_2nd'].set_params(
            #     solver='lbfgs', penalty='l2', multi_class='multinomial')

        # RandomForestClf better suited to handle lot of categories
        if labels is not None and len(labels) > 10:
            del models_and_params['GBoostingClf_2nd']

    return models_and_params


def models_and_params_for_nested_cv_tasks(y_type, scoring, labels):
    models_and_params = dict()

    if y_type == 'binary':
            models_and_params = pgd.full_search_models_and_parameters
            models_and_params['LogRClf_2nd'][0].set_params(solver='liblinear')

    else:
        # y_type == 'multiclass'

        if scoring == 'neg_log_loss':
            models_and_params = dict()

            for k, v in pgd.full_search_models_and_parameters.items():
                if hasattr(v[0], 'predict_proba'):
                    models_and_params[k] = v

            # solver='saga', penalty='l1'
            models_and_params['LogRClf_2nd'][0].set_params(
                solver='lbfgs', penalty='l2', multi_class='multinomial')

        # RandomForestClf better suited to handle lot of categories
        if labels is not None and len(labels) > 10:
            del models_and_params['GBoostingClf_2nd']

    return models_and_params


def create_ensemble_of_best_models(best_model_name, best_model_estim, seed=0):
    base_estimator_name = best_model_name.strip('_2nd')
    bagging_estimator_name = 'BaggingClf_2nd_' + base_estimator_name

    bagging_param_grid = {
        bagging_estimator_name + '__'
        + k: v for k, v in pgd.Bagging_param_grid.items()
        }
    bagging = BaggingClassifier(best_model_estim, random_state=seed)

    return bagging, bagging_estimator_name, bagging_param_grid


def perform_classic_cv_evaluation_and_calibration(
        auto_feat_eng_data, scoring, Y_type, labels=None, 
        d_name=None, random_state=0):
    """
    # perform non-nested cross validation and calibration of best estimator.

    ---------------------------------------------------------
    auto_feat_eng_data:
        dictionary with encoder, scaler, feature selector,
        Pipeline.steps and train-test data
    scoring: scoring for model evaluation
        -- string ('roc_auc_score') or list of strings
    random_state: seed
    """
    if isinstance(scoring, list) or isinstance(scoring, dict) or isinstance(
                  scoring, tuple):
        raise TypeError("""
        'perform_classic_cv_evaluation_and_calibration' method allows only
        to perform single-metric evaluations.
        """)
    if scoring not in (None, 'accuracy', 'roc_auc', 'neg_log_loss'):
        raise ValueError("""
        %s is not a valid scoring value
        for method 'perform_classic_cv_evaluation_and_calibration'.
        Valid options are ['accuracy', 'roc_auc', 'neg_log_loss']
        """ % scoring)

    print("### Probability Calibration of Best Estimator "
          "-- CalibratedClassifierCV with cv=cv (no prefit) ###")
    print("### Non-nested Cross-Validation ###")
    print("Models are trained and calibrated on the same data, train data,\n"
          "calibration is evaluated on test data. No nested-cv is performed.")
    print()
    print("pipe.fit(train_data)")

    encoding = auto_feat_eng_data['encoding']
    scaler_tuple = auto_feat_eng_data['scaler']
    featselector = auto_feat_eng_data['feat_selector']
    steps = auto_feat_eng_data['steps']
    X_train_transformed, y_train, X_test_transformed, y_test = auto_feat_eng_data['data_arrays']
    X, y = auto_feat_eng_data['Xy']

    print()

    print()
    print("X_train shape: ", X_train_transformed.shape)
    # print("X_train -- first row:", X_train.values[0])
    print("X_train -- first row:", X_train_transformed[0])
    print("y_train shape: ", y_train.shape)
    print()

    print("X_test shape: ", X_test_transformed.shape)
    # print("X_test -- first row:", X_test.values[0])
    print("X_test -- first row:", X_test_transformed[0])
    print("y_test shape: ", y_test.shape)
    print()

    # input("Press any key to continue...")

    # Start evaluation process

    # Evaluation of best model with non-nested CV -- outer: CV

    n_splits = 3   # au.select_nr_of_splits_for_kfold_cv()

    # dict of models and their associated parameters
    # if it comes out that the best model is LogReg, no comparison is needed

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

    print()
    print("=== [task] Evaluation of DummyClassifier")
    print()

    wtr = eu.calculate_sample_weight(y_train)

    print("=== 'sample_weight'")
    print(wtr[:5])
    print("=== target train data sample")
    print(y_train[:5])
    print()

    # This cross-validation object is
    # a variation of KFold that returns stratified folds.
    # The folds are made by preserving
    # the percentage of samples for each class.
    outer_cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)

    strategy = 'stratified' # 'most_frequent'

    average_scores_and_best_scores = eu.single_classic_cv_evaluation(
        X_train_transformed, y_train, 'DummyClf_2nd',
        DummyClassifier(strategy=strategy), wtr, scoring, outer_cv,
        dict(), scores_of_best_model, results, names, random_state)

    scores_of_best_model = average_scores_and_best_scores[1]

    Dummy_scores.append(scores_of_best_model[0])   # Dummy score -- ROC_AUC
    Dummy_scores.append(scores_of_best_model[1])   # Dummy score std
    Dummy_scores.append(scores_of_best_model[2])   # Dummy cv results for score
    # Dummy_scores.append(scores_of_best_model[2]) # Dummy Brier score loss
    Dummy_scores.append(scores_of_best_model[3])   # Dummy execution time
    # Dummy model's name and estimator
    Dummy_scores.append(scores_of_best_model[4])

    names = []
    results = []

    # Non-Nested CV: its purpose is to estimate how well
    # models would perform with their default parameters already tuned
    # or with default parameters

    print()
    print("=== Classic Cross-Validation")
    print()

    print("Estimators before model evaluation:", steps)
    print()

    # we will collect the average of the scores on the k outer folds in
    # this dictionary with keys given by the names of the models
    # in 'pgd.starting_point_models_and_params'
    average_scores_across_outer_folds_for_each_model = dict()

    # this holds for regression as well, not time-series

    models_and_parameters = models_and_params_for_classic_cv_tasks(
        Y_type, scoring, labels)

    average_scores_and_best_scores = eu.classic_cv_model_evaluation(
        X_train_transformed, y_train, models_and_parameters,
        # {},
        scoring, outer_cv, average_scores_across_outer_folds_for_each_model,
        scores_of_best_model, results, names, random_state)

    print()
    au.box_plots_of_models_performance(results, names)

    print()
    # input("Press any key to continue...")

    results = []
    names = []

    print()
    print("=== After Classic CV evaluation...")
    print()

    average_scores_across_outer_folds_for_each_model = average_scores_and_best_scores[0]
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
    print("... execution time: %.2fs" % best_exec_time)
    # print("and prediction confidence: %1.3f" % best_brier_score)
    print()

    print("=== Classic CV to evaluate more complex models")
    print()

    complex_models_and_parameters = dict()
    average_scores_across_outer_folds_complex = dict()

    all_models_and_parameters = models_and_parameters

    # Let's add some simple neural network

    print("=== [task] Comparing best model to simple Neural Network "
          "(with single or two hidden layers).")
    print()

    # This is an experiment to check 
    # how different Keras architectures perform
    # to avoid hard-coding NNs, you should determine at least 
    # nr of layers and nr of nodes by using Grid or Randomized Search CV

    input_dim = int(X_train_transformed.shape[1])

    nb_epoch = au.select_nr_of_iterations('nn')

    batch_size = 32

    kclf_names_and_models = create_keras_classifiers(
        Y_type, input_dim, labels, nb_epoch, batch_size)

    for k, v in kclf_names_and_models.items():
        complex_models_and_parameters[k] = v

    average_scores_and_best_scores = eu.classic_cv_model_evaluation(
        X_train_transformed, y_train, complex_models_and_parameters, scoring,
        outer_cv, average_scores_across_outer_folds_complex,
        scores_of_best_model, results, names, random_state)

    print()
    au.box_plots_of_models_performance(results, names)

    print()
    print("=== After Classic CV evaluation of complex models...")
    print()

    average_scores_across_outer_folds_for_each_model = average_scores_and_best_scores[0]
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

        preprocessing = (encoding, scaler_tuple, featselector)

        if labels is not None:
            if not isinstance(labels, list):
                raise TypeError("Multiclass models need a list of string labels.")
            else:
                print("You have labels:", labels)
                all_models_and_parameters['labels'] = labels

        print("Defined dictionary with models, parameters and related data.")
        print()

        tc.calibrate_best_model(
            X, y, X_train_transformed, X_test_transformed,
            y_train, y_test, auto_feat_eng_data['tt_index'], 
            preprocessing, scores_of_best_model,
            all_models_and_parameters, n_splits, nb_epoch,
            scoring, models_data, d_name, random_state)
    else:
        sys.exit("Your best classifier is not a good classifier.")


def perform_nested_cv_evaluation_and_calibration(
        auto_feat_eng_data, nested_cv_scoring, Y_type, labels=None,
        d_name=None, random_state=0, followup=False):
    """
    # perform nested cross validation and calibration of best estimator.

    ---------------------------------------------------------
    auto_feat_eng_data:
        dictionary with encoder, scaler, feature selector,
        Pipeline.steps and train-test data
    scoring: scoring for model evaluation
        -- string ('roc_auc_score') or list of strings
    random_state: seed
    """
    if isinstance(nested_cv_scoring, list) or isinstance(
            nested_cv_scoring, dict) or isinstance(nested_cv_scoring, tuple):
        raise TypeError("""
            'perform_nested_cv_evaluation_and_calibration' method allows only
            to perform single-metric evaluations.
            """)
    if nested_cv_scoring not in (None, 'accuracy', 'roc_auc', 'neg_log_loss'):
        raise ValueError("""
        %s is not a valid nested_cv_scoring value for method
        'perform_nested_cv_evaluation_and_calibration'. Valid options are
        ['accuracy', 'roc_auc', 'neg_log_loss']""" % nested_cv_scoring)

    print("### Probability Calibration of Best Estimator "
          "-- CalibratedClassifierCV with cv=cv (no prefit) ###")
    print("### Nested Cross-Validation ###")
    print("Models are trained and calibrated on the same data, train data,\n"
          "calibration is evaluated on test data.")
    print()
    print("RSCV.refit=False")
    print("pipe.fit(train_data)")

    # each one of these items need a check
    encoding = auto_feat_eng_data['encoding']
    scaler_tuple = auto_feat_eng_data['scaler']
    featselector = auto_feat_eng_data['feat_selector']
    steps = auto_feat_eng_data['steps']
    X_train_transformed, y_train, X_test_transformed, y_test = auto_feat_eng_data['data_arrays']
    X, y = auto_feat_eng_data['Xy']
    train_index, test_index = auto_feat_eng_data['tt_index']

    print()
    if followup:
        print("X_train_transformed shape: ", X_train_transformed.shape)
        print("X_test_transformed shape: ", X_test_transformed.shape)
    
    n_splits = au.select_nr_of_splits_for_kfold_cv()

    n_iter = au.select_nr_of_iterations()

    # Stratified folds preserve the percentage of samples for each class.
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)

    ###

    # you should check Y for categorical values and
    # eventually label encode them...

    # Nested [RSCV] CV

    print()

    print("Metric:", nested_cv_scoring)
    print("Calibration of untrained models -- CCCV 2nd")
    print()

    # Evaluation of best modelwith nested CV -- inner: RSCV

    # if it comes out that the best model is LogReg, no comparison is needed

    best_atts = eu.best_model_initial_attributes(nested_cv_scoring, n_splits)

    best_score, best_score_dev, best_cv_results, best_model_name = best_atts

    best_exec_time = 31536000    # one year in seconds
    best_model = (best_model_name, None, None)

    Dummy_scores = []

    models_data = []
    names = []
    results = []

    scores_of_best_model = (best_score, best_score_dev, best_cv_results,
                            best_exec_time, best_model)

    # Start evaluation process

    print()
    print("=== [task] Evaluation of DummyClassifier")
    print()

    wtr = eu.calculate_sample_weight(y_train)

    print("=== 'sample_weight'")
    print(wtr[:5])
    print("=== target train data sample")
    print(y_train[:5])
    print()

    strategy = 'stratified' # 'most_frequent'

    average_scores_and_best_scores = eu.single_nested_rscv_evaluation(
        X_train_transformed, y_train, 'DummyClf_2nd',
        DummyClassifier(strategy=strategy), dict(), wtr,
        nested_cv_scoring, 0, inner_cv, outer_cv, dict(), scores_of_best_model,
        results, names, random_state)

    scores_of_best_model = average_scores_and_best_scores[1]

    Dummy_scores.append(scores_of_best_model[0])    # Dummy score -- ROC_AUC
    Dummy_scores.append(scores_of_best_model[1])    # Dummy score std
    Dummy_scores.append(scores_of_best_model[2])    # Dummy cv results
    Dummy_scores.append(scores_of_best_model[3])    # Dummy execution time
    # Dummy model's name and estimator
    Dummy_scores.append(scores_of_best_model[4])

    names = []
    results = []

    # Nested CV: its purpose is not to find best parameters, but
    # how well models would perform with their parameters tuned

    print()
    print("=== Nested CV [inner cv: RSCV]")
    print()

    # we will collect the average of the scores on the k outer folds in
    # this dictionary with keys given by the names of the models
    # in 'models_and_parameters'
    average_scores_across_outer_folds_for_each_model = dict()

    # this holds for regression

    # update models_and_params dict according
    # to learning mode 'quick', 'standard', 'hard'

    models_and_parameters = models_and_params_for_nested_cv_tasks(
        Y_type, nested_cv_scoring, labels)

    average_scores_and_best_scores = eu.nested_rscv_model_evaluation(
            X_train_transformed, y_train, models_and_parameters,
            # {},
            nested_cv_scoring, n_iter, inner_cv, outer_cv,
            average_scores_across_outer_folds_for_each_model,
            scores_of_best_model, results, names, random_state)

    print()
    au.box_plots_of_models_performance(results, names)

    results = []
    names = []

    print()
    print("=== After Nested CV evaluation...")
    print()

    average_scores_across_outer_folds_for_each_model = average_scores_and_best_scores[0]
    scores_of_best_model = average_scores_and_best_scores[1]

    best_model_name = scores_of_best_model[4][0]
    best_model_estim = scores_of_best_model[4][1]
    # no need to define a Keras build function here
    # best_nn_build_fn = scores_of_best_model[3][2]
    best_score = scores_of_best_model[0]
    best_score_dev = scores_of_best_model[1]
    best_cv_results = scores_of_best_model[2]
    best_exec_time = scores_of_best_model[3]

    print()
    print("Currently, best model is '%s' with score '%s': %1.3f (%1.3f)... :"
          % (best_model_name, nested_cv_scoring.strip('neg_'), best_score,
             best_score_dev))
    print("... execution time: %.2fs" % best_exec_time)
    print()
    print()

    print("======= Nested RSCV to evaluate more complex models")
    print()

    # complex_models_and_parameters[name] = (model, rscv_params, dict())
    complex_models_and_parameters = dict()
    average_scores_across_outer_folds_complex = dict()

    # all_models_and_parameters = {}
    all_models_and_parameters = models_and_parameters

    if labels is not None:
        if not isinstance(labels, list):
            raise TypeError("Multiclass models need a list of string labels.")
        else:
            print("You have labels:", labels)
            all_models_and_parameters['labels'] = labels

    print("Defined dictionary with models, parameters and related data.")
    print()

    # Compare to ensemble of instances of best model
    # after looping over standard models and once you have best model,
    # create ensemble of it

    bagging_estimator_name = ''

    if best_model_name not in {
        'DecisionTreeClf_2nd', 'ExtraTreesClf_2nd', 'RandomForestClf_2nd',
            'GBoostingClf_2nd', 'XGBClf_2nd', 'AdaBClf_2nd',
            'Bagging_SVMClf_2nd'}:

        print("=== [task] Comparing best model to ensemble of instances "
              "of best model:")
        print("BaggingClf(%s)" % best_model_name)
        print()

        # steps = []
        # ...
        # steps.append(('feature_union', feature_union))

        bag_estim, bag_estim_name, bag_param_grid = create_ensemble_of_best_models(
            best_model_name, best_model_estim, random_state)

        # add bagging to dictionary of complex models

        complex_models_and_parameters[bag_estim_name] = (
            bag_estim, bag_param_grid
            )

        all_models_and_parameters[bag_estim_name] = (
            bag_estim, bag_param_grid
            )

    # Let's add some simple neural network

    print("=== [task] Comparing best model to simple Neural Network "
          "(with single or two hidden layers).")
    print()

    input_dim = int(X_train_transformed.shape[1])

    if not followup:
        nb_epoch = au.select_nr_of_iterations('nn')
    else:
        nb_epoch = au.select_nr_of_iterations('nn', followup)

    keras_clf_name = "KerasClf_2nd"

    keras_nn_model, keras_param_grid = create_best_keras_clf_architecture(
        keras_clf_name, Y_type, labels, input_dim, nb_epoch, pgd.Keras_param_grid)

    complex_models_and_parameters[keras_clf_name] = (
        keras_nn_model, keras_param_grid)

    # Feed nested-cv function with dictionary of models and their params

    average_scores_and_best_scores_complex = eu.nested_rscv_model_evaluation(
        X_train_transformed, y_train, complex_models_and_parameters,
        nested_cv_scoring, n_iter, inner_cv, outer_cv,
        average_scores_across_outer_folds_complex, scores_of_best_model,
        results, names, random_state)

    print()
    au.box_plots_of_models_performance(results, names)

    print()
    print("=== After Nested CV evaluation of complex models...")
    print()

    average_scores_across_outer_folds_complex =\
    average_scores_and_best_scores_complex[0]
    scores_of_best_model = average_scores_and_best_scores_complex[1]

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

    print("Currently, best model is '%s' with score '%s': %1.3f (%1.3f)... :"
          % (best_model_name, nested_cv_scoring, best_score, best_score_dev))
    if best_model_name == keras_clf_name:
        best_nn_build_fn = scores_of_best_model[4][2]
        print("Best build function:", best_nn_build_fn)
    print("... execution time: %.2fs" % best_exec_time)
    print()

    if best_model_name != 'DummyClf_2nd':
        # best model is supposed to have passed some statistical test
        print("DummyClassifier's scores -- '%s': %1.3f (%1.3f)" % (
            nested_cv_scoring, Dummy_score, Dummy_score_dev))
        print("'%s' does better than DummyClassifier." % best_model_name)
        print("Execution time of '%s': %.2fs" % (
            best_model_name, best_exec_time))
        if best_exec_time < Dummy_exec_time:
            print("'%s' is quicker than DummyClf." % best_model_name)

        print()
        print()
        # input("Press key to continue...")

        preprocessing = (encoding, scaler_tuple, featselector)

        tc.tune_calibrate_best_model(
            X, y, X_train_transformed, X_test_transformed,
            y_train, y_test, auto_feat_eng_data['tt_index'], 
            preprocessing, scores_of_best_model,
            all_models_and_parameters, n_splits, n_iter, nb_epoch,
            nested_cv_scoring, models_data, d_name, random_state)

    else:
        sys.exit("Your best classifier is not a good classifier.")


# select your fraking strategy
def select_evaluation_strategy(
        auto_feat_eng_data, target, test_frac=0.3,
        odf=None, scoring='roc_auc', Y_type='binary', labels=None,
        d_name=None, random_state=0, learn='standard', mode='interactive'):
    """
    # select evaluation strategy.

    ----------------------------------------------------------------------------
    'auto_feat_eng_data': engineered data from split_and_X_encode function

    'target': label # or single column dataframe with labels

    'odf' : orginal dataframe with eventual feature engineering
        not causing data leakage

    'scoring' : scoring for model evaluation

    'Y_type' : type of target ; default: 'binary'

    'labels' : labels for multiclass logistic regression metric

    'random_state' : random state (seed)

    'learn' : learn mode based on output of learning_strategy() fct

    'mode' : model of machine learnin problem solution;
         default: 'interactive', else 'auto'
    """
    if learn == 'quick':

        perform_classic_cv_evaluation_and_calibration(
            auto_feat_eng_data, scoring, Y_type, labels, 
            d_name, random_state)

        msg = "Are you satisfied with current results?"

        if au.say_yes_or_no(msg) in {"YES", "yes", "Y", "y"}:
            print("Great! See you next time!")
            print()
        else:

            if odf is not None:  # df has been reduced in size

                msg = "Do you want to use the full dataset?"

                if au.say_yes_or_no(msg) in {"YES", "yes", "Y", "y"}:

                    print()
                    print("### Split and encode the whole dataframe")
                    print("Warning! No feature engineering implemented!")
                    print()

                    split_enc_X_data = eu.split_and_X_encode(
                        odf, target, test_frac, random_state)

                    auto_feat_eng_data, scoring, Y_type, classes = split_enc_X_data

                else:
                    print("### Use current data from small/smaller dataframe")
            
            else:  # df has not been reduced in size, full df
                # you already did you feature engineering

                print()
                print("### Using the whole dataframe")
                print()

            perform_nested_cv_evaluation_and_calibration(
               auto_feat_eng_data, scoring, Y_type, labels, 
               d_name, random_state, True)

    elif learn == 'standard':

        perform_nested_cv_evaluation_and_calibration(
             auto_feat_eng_data, scoring, Y_type, labels, d_name, random_state)

    else:
        # learn == 'large'
        perform_classic_cv_evaluation_and_calibration(
             auto_feat_eng_data, scoring, Y_type, labels, d_name,
             random_state)
