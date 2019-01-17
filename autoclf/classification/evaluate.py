"""
This module contains functions to perform non-nested and nested cv.

This module contains a function to select a strategy for models evaluation.

"""

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from . import param_grids_distros as pgd
from . import neuralnets as nn
from operator import itemgetter

import numpy as np
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

    models_and_parameters = dict()

    if Y_type == 'binary':
        models_and_parameters = pgd.starting_point_models_and_params
        # models_and_parameters['LogRClf_2nd'].set_params(
        #     solver='liblinear')

    else:
        # if Y_type == 'multiclass':

        if scoring == 'neg_log_loss':               
            for k, v in pgd.starting_point_models_and_params.items():
                if hasattr(v, 'predict_proba'):
                    print(k)
                    models_and_parameters[k] = v

            # solver='saga', penalty='l1'
            # models_and_parameters['LogRClf_2nd'].set_params(
            #     solver='lbfgs', penalty='l2', multi_class='multinomial')

        # RandomForestClf better suited to handle lot of categories
        if len(labels) > 10:
            del models_and_parameters['GBoostingClf_2nd']

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
    # no need to define a Keras build function here
    # best_nn_build_fn = scores_of_best_model[4][2]
    best_score = scores_of_best_model[0]
    best_score_dev = scores_of_best_model[1]
    # best_brier_score = scores_of_best_model[2]
    best_cv_results = scores_of_best_model[2]
    best_exec_time = scores_of_best_model[3]

    print()
    print("Currently, best model is '%s' with score '%s': %1.3f (%1.3f)... :"
          % (best_model_name, scoring.strip('neg_'), best_score,
             best_score_dev))
    print("... execution time: %.2fs" % best_exec_time)
    # print("and prediction confidence: %1.3f" % best_brier_score)
    print()
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

    output_dim = 1

    batch_size = 32

    if Y_type == 'multiclass':

        output_dim = len(labels)

        baseline_nn_default = KerasClassifier(
            build_fn=nn.baseline_nn_model_multilabel, nb_epoch=nb_epoch,
            input_dim=input_dim, output_dim=output_dim, batch_size=batch_size,
            verbose=0
            )

        complex_models_and_parameters['baseline_nn_default_Clf_2nd'] = baseline_nn_default

        # build smaller layer

        baseline_nn_smaller = KerasClassifier(
            build_fn=nn.baseline_nn_model_multilabel, nb_epoch=nb_epoch,
            input_dim=input_dim, output_dim=output_dim, batch_size=batch_size,
            verbose=0
            )

        complex_models_and_parameters['baseline_nn_smaller_Clf_2nd'] = baseline_nn_smaller

        # build larger layer

        larger_nn = KerasClassifier(
            build_fn=nn.larger_nn_model_multilabel, nb_epoch=nb_epoch,
            input_dim=input_dim, output_dim=output_dim, batch_size=batch_size,
            verbose=0
            )

        complex_models_and_parameters['larger_nn_Clf_2nd'] = larger_nn

        # shallow deep nn

        small_deep_nn = KerasClassifier(
            build_fn=nn.deep_nn_model_multilabel, nb_epoch=nb_epoch,
            input_dim=input_dim, output_dim=output_dim, batch_size=batch_size,
            verbose=0
            )

        complex_models_and_parameters['deep_nn_Clf_2nd'] = small_deep_nn

        # deeper deep nn

        if input_dim < 15:
            deep_nn = KerasClassifier(
                build_fn=nn.deeper_nn_model_multilabel, nb_epoch=nb_epoch,
                input_dim=input_dim, output_dim=output_dim,
                batch_size=batch_size, verbose=0
                )

            complex_models_and_parameters['deeper_nn_Clf_2nd'] = deep_nn

    else:

        # you could grid search nn_model's parameters space using hyperas...
        # you need KerasClassifier wrapper to use Keras models in sklearn

        baseline_nn_default = KerasClassifier(
            build_fn=nn.baseline_nn_model, nb_epoch=nb_epoch,
            input_dim=input_dim, batch_size=batch_size, verbose=0
            )

        complex_models_and_parameters['baseline_nn_default_Clf_2nd'] = baseline_nn_default

        # build smaller layer

        baseline_nn_smaller = KerasClassifier(
            build_fn=nn.baseline_nn_model, nb_epoch=nb_epoch,
            input_dim=input_dim, batch_size=batch_size, verbose=0
            )

        complex_models_and_parameters['baseline_nn_smaller_Clf_2nd'] = baseline_nn_smaller

        # build larger layer

        larger_nn = KerasClassifier(
            build_fn=nn.larger_nn_model, nb_epoch=nb_epoch,
            input_dim=input_dim, batch_size=batch_size, verbose=0
            )

        complex_models_and_parameters['larger_nn_Clf_2nd'] = larger_nn

        # shallow deep nn

        small_deep_nn = KerasClassifier(
            build_fn=nn.deep_nn_model, nb_epoch=nb_epoch, input_dim=input_dim,
            batch_size=batch_size, verbose=0
            )

        complex_models_and_parameters['deep_nn_Clf_2nd'] = small_deep_nn

        # deeper deep nn

        deep_nn = KerasClassifier(
            build_fn=nn.deeper_nn_model, nb_epoch=nb_epoch,
            input_dim=input_dim, batch_size=batch_size, verbose=0
            )

        complex_models_and_parameters['deeper_nn_Clf_2nd'] = deep_nn

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

    print()
    print("Currently, best model is '%s' with score '%s': %1.3f (%1.3f)... :" %
          (best_model_name, scoring.strip('neg_'), best_score, best_score_dev))
    if best_model_name in (
            'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
            'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'deeper_nn_Clf_2nd'):
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
        # input("Press key to continue...")

        preprocessing = (encoding, scaler_tuple, featselector)

        if labels is not None:
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

    encoding = auto_feat_eng_data['encoding']
    scaler_tuple = auto_feat_eng_data['scaler']
    featselector = auto_feat_eng_data['feat_selector']
    steps = auto_feat_eng_data['steps']
    X_train_transformed, y_train, X_test_transformed, y_test = auto_feat_eng_data['data_arrays']
    X, y = auto_feat_eng_data['Xy']
    train_index, test_index = auto_feat_eng_data['tt_index']

    print()
    
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

    if Y_type == 'binary':
        models_and_parameters = pgd.full_search_models_and_parameters
        models_and_parameters['LogRClf_2nd'][0].set_params(
            solver='liblinear')

    else:
        # Y_type == 'multiclass'

        if nested_cv_scoring == 'neg_log_loss':
            models_and_parameters = dict()

            for k, v in pgd.full_search_models_and_parameters.items():
                if hasattr(v[0], 'predict_proba'):
                    models_and_parameters[k] = v

            # solver='saga', penalty='l1'
            models_and_parameters['LogRClf_2nd'][0].set_params(
                solver='lbfgs', penalty='l2', multi_class='multinomial')

        # RandomForestClf better suited to handle lot of categories
        if len(labels) > 10:
            del models_and_parameters['GBoostingClf_2nd']

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

        base_estimator = best_model_estim
        base_estimator_name = best_model_name.strip('_2nd')
        bagging_estimator_name = 'BaggingClf_2nd_' + base_estimator_name

        bagging_param_grid = {
            bagging_estimator_name + '__'
            + k: v for k, v in pgd.Bagging_param_grid.items()
            }
        bagging = BaggingClassifier(base_estimator, random_state=random_state)

        # add bagging to dictionary of complex models

        complex_models_and_parameters[bagging_estimator_name] = (
            bagging, bagging_param_grid
            )

        all_models_and_parameters[bagging_estimator_name] = (
            bagging, bagging_param_grid
            )

    # Compare to VotingClassifier

    print("=== [task] Comparing best model to VotingClassifier "
          "taking top 3 models.")
    print()

    # average_scores_across_outer_folds[name] = (
    #     score, score_dev, exec_time, model, params)

    # dictionary of top 3 classifiers having 'predict_proba' attribute

    candidate_estimators_for_vc3 = dict()

    for k, v in average_scores_across_outer_folds_for_each_model.items():
        if hasattr(v[3], 'predict_proba'):
            candidate_estimators_for_vc3[k] = v
            print("Model '%s' [%1.3f (%1.3f)] has attribute 'predict_proba'."
                  % (k, v[0], v[1]))

    print()

    if 'Bagging_SVMClf_2nd' in candidate_estimators_for_vc3:
        # this is gonna slow or break evaluation of VotingClf
        del candidate_estimators_for_vc3['Bagging_SVMClf_2nd']

    if nested_cv_scoring not in ('neg_log_loss', 'brier_score_loss'):
        reverse = True
    else:
        reverse = False

    candidate_estimators_for_vc3 = sorted(
        candidate_estimators_for_vc3.items(),
        key=itemgetter(1), reverse=reverse
        )

    print()
    print("Candidate estimators for VotingClf3")
    print(candidate_estimators_for_vc3)

    # you should sort based on brier -- the lower, the better
    top_3_models_data = dict(
        # sort based on score: key=itemgetter(1)
        sorted(candidate_estimators_for_vc3, key=itemgetter(1))[:3]
        )

    vc3_estimator_name = 'VClf_3_2nd'
    vc3_estimators = []

    if len(top_3_models_data) == 3:

        # merge top 3 models param dictionaries

        all_models_and_parameters[vc3_estimator_name] = (
            top_3_models_data, dict()
            )

        nr_clf = 1
        for k, v in top_3_models_data.items():
            print("Estimator nr. %d: %s" % (nr_clf, k))
            p = Pipeline([(k, v[3])])
            vc3_estimators.append(('p' + str(nr_clf), p))
            nr_clf += 1

        # print("VClf3 estimators -- top 3 models':", vc3_estimators)
        # print()

        list_of_params = [v[4] for k, v in top_3_models_data.items()]

    else:
        print()
        print("Could not retrieve 3 estimators for VotingClassifier.")
        print("Using fixed pools of pre-determined estimators.")
        print()

        vc3_estimators_data = dict()

        vc3_estimators_data['LogRClf_2nd'] = (
            models_and_parameters['LogRClf_2nd'][0],
                models_and_parameters['LogRClf_2nd'][1])

        vc3_estimators_data['GaussianNBClf_2nd'] = (
            models_and_parameters['GaussianNBClf_2nd'][0],
                models_and_parameters['GaussianNBClf_2nd'][1])

        vc3_estimators_data['RandomForestClf_2nd'] = (
            models_and_parameters['RandomForestClf_2nd'][0],
                models_and_parameters['RandomForestClf_2nd'][1])

        all_models_and_parameters[vc3_estimator_name] = (
            vc3_estimators_data, dict()
            )

        print("vc3_estimators_data:\n", vc3_estimators_data)

        nr_clf = 1
        for k, v in vc3_estimators_data.items():
            print("Estimator nr. %d: %s" % (nr_clf, k))
            p = Pipeline([(k, v[0])])
            vc3_estimators.append(('p' + str(nr_clf), p))
            nr_clf += 1

        list_of_params = [v[1] for k, v in vc3_estimators_data.items()]

    print("List of VClf3 params:", list_of_params)
    print()
    # input("Press any key to continue...")

    vc3_params = dict()

    nr_clf = 1
    for par in list_of_params:
        new_par = {vc3_estimator_name + '__p' + str(nr_clf)
                   + '__' + k: v for k, v in par.items()}
        vc3_params.update(new_par)
        nr_clf += 1

    vc3_param_grid = {
        vc3_estimator_name + '__'
        + k: v for k, v in pgd.VC_3_param_grid.items()}

    vc3_params.update(vc3_param_grid)

    # print("Dict of VClf3 params:", vc3_params)

    # top 3 estimators here should be well calibrated
    vclf3 = VotingClassifier(vc3_estimators, voting='soft')

    # add votingclf of top 3 to dictionary of complex models

    complex_models_and_parameters[vc3_estimator_name] = (vclf3, vc3_params)

    print()

    # Let's add some simple neural network

    print("=== [task] Comparing best model to simple Neural Network "
          "(with single or two hidden layers).")
    print()

    input_dim = int(X_train_transformed.shape[1])
    if not followup:
        nb_epoch = au.select_nr_of_iterations('nn')
    else:
        nb_epoch = au.select_nr_of_iterations('nn', followup)

    batch_size = 32

    if Y_type == 'multiclass':

        output_dim = len(labels)

        baseline_nn_default = KerasClassifier(
            build_fn=nn.baseline_nn_model_multilabel, nb_epoch=nb_epoch,
            input_dim=input_dim, output_dim=output_dim,
            batch_size=batch_size,
            verbose=0
        )

        complex_models_and_parameters['baseline_nn_default_Clf_2nd'] = (
            baseline_nn_default, dict())

        # build smaller layer

        baseline_nn_smaller = KerasClassifier(
            build_fn=nn.baseline_nn_model_multilabel, nb_epoch=nb_epoch,
            input_dim=input_dim, output_dim=output_dim,
            batch_size=batch_size, verbose=0
            )

        complex_models_and_parameters['baseline_nn_smaller_Clf_2nd'] = (
            baseline_nn_smaller, dict())

        # build larger layer

        larger_nn = KerasClassifier(
            build_fn=nn.larger_nn_model_multilabel, nb_epoch=nb_epoch,
            input_dim=input_dim, output_dim=output_dim, batch_size=batch_size, verbose=0
            )

        complex_models_and_parameters['larger_nn_Clf_2nd'] = (
            larger_nn, dict())

        # shallow deep nn

        small_deep_nn = KerasClassifier(
            build_fn=nn.deep_nn_model_multilabel, nb_epoch=nb_epoch,
            input_dim=input_dim, output_dim=output_dim, batch_size=batch_size, verbose=0
            )

        complex_models_and_parameters['deep_nn_Clf_2nd'] = (
            small_deep_nn, dict())

        # deeper deep nn

        deep_nn = KerasClassifier(
            build_fn=nn.deeper_nn_model_multilabel, nb_epoch=nb_epoch,
            input_dim=input_dim, output_dim=output_dim, batch_size=batch_size, verbose=0
            )

        complex_models_and_parameters['deeper_nn_Clf_2nd'] = (deep_nn, dict())

    else:

        # you could grid search nn_model's parameters space using hyperas...
        # you need KerasClassifier wrapper to use Keras models in sklearn

        baseline_nn_default = KerasClassifier(
            build_fn=nn.baseline_nn_model, nb_epoch=nb_epoch,
            input_dim=input_dim, batch_size=batch_size, verbose=0
            )

        complex_models_and_parameters['baseline_nn_default_Clf_2nd'] = (
            baseline_nn_default, dict())

        # build smaller layer

        baseline_nn_smaller = KerasClassifier(
            build_fn=nn.baseline_nn_model, nb_epoch=nb_epoch,
            input_dim=input_dim, batch_size=batch_size, verbose=0
            )

        complex_models_and_parameters['baseline_nn_smaller_Clf_2nd'] = (
            baseline_nn_smaller, dict())

        # build larger layer

        larger_nn = KerasClassifier(
            build_fn=nn.larger_nn_model, nb_epoch=nb_epoch,
            input_dim=input_dim, batch_size=batch_size, verbose=0
            )

        complex_models_and_parameters['larger_nn_Clf_2nd'] = (
            larger_nn, dict())

        # shallow deep nn

        small_deep_nn = KerasClassifier(
            build_fn=nn.deep_nn_model, nb_epoch=nb_epoch, input_dim=input_dim,
            batch_size=batch_size, verbose=0
            )

        complex_models_and_parameters['deep_nn_Clf_2nd'] = (
            small_deep_nn, dict())

        # deeper deep nn

        deep_nn = KerasClassifier(
            build_fn=nn.deeper_nn_model, nb_epoch=nb_epoch, input_dim=input_dim,
            batch_size=batch_size, verbose=0
            )

        complex_models_and_parameters['deeper_nn_Clf_2nd'] = (deep_nn, dict())

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
    if best_model_name in (
        'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
            'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'deeper_nn_Clf_2nd'):
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

            if odf is not None:

                print("### Split and encode the whole dataframe")

                split_enc_X_data = eu.split_and_X_encode(
                     odf, target, test_frac, random_state)

                auto_feat_eng_data, scoring, Y_type, classes = split_enc_X_data

            else:
                print("### Use current data from small/smaller dataframe")

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
