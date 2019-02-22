"""

This module allows you to tune and calibrate the best estimator.

This module allows you to tune and calibrate the best estimator
returned by nested cv evaluation, and to just calibrate
the best estimator returned by the non-nested cv evaluation.

"""

from sklearn.calibration import CalibratedClassifierCV

import numpy as np
import re
from random import randint
from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt

from . import param_grids_distros as pgd
from .. import auto_utils as au
from . import eval_utils as eu

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import multiclass as mc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.wrappers.scikit_learn import KerasClassifier
sys.stderr = stderr

import warnings
warnings.filterwarnings("ignore")

def best_keras_clf_estimator(
    y_type, best_nn_build_fn, nb_epoch, input_dim, labels, batch_size=None):

    best_model_estim = None

    best_model_estim = KerasClassifier(
        build_fn=best_nn_build_fn, nb_epoch=nb_epoch,
        input_dim=input_dim, verbose=0)

    if y_type == 'multiclass':

        if labels is None:
            raise ValueError("%r is not a valid type for var 'labels'" % labels)
        elif not isinstance(labels, list):
            raise TypeError("Multiclass keras models need a list of string labels.")
        else:
            output_dim = len(labels)

        best_model_estim.set_params(output_dim=output_dim)

    if batch_size is not None and isinstance(batch_size, int):
        best_model_estim.set_params(batch_size=batch_size)

    return best_model_estim


def confusion_matrix_and_clf_report(y_type, model_name, y_test, y_pred):
    if y_type == 'binary':
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print()
        print("Test errors for '%s'" % model_name)
        print("\ttrue negatives: %d" % tn)
        print("\tfalse positives: %d" % fp)
        print("\tfalse negatives: %d" % fn)
        print("\ttrue positives: %d" % tp)
    else:
        print()
        print("Confusion matrix for '%s'.\n"
                % model_name, confusion_matrix(y_test, y_pred))
        print()
    print("Classification report for '%s'\n"
            % model_name, classification_report(y_test, y_pred))


def check_model_hasroc(model_name, model_data):

    has_roc = 0
    try:
        model_data[2]
    except IndexError:
        print("ROC_AUC score is not available.")
        roc_auc = None
    except Exception as e:
        print(e)
    else:
        roc_auc = model_data[2]
        has_roc = 1
    finally:
        print(
            "We have %s data to compare prediction confidence of models." 
            % model_name)
        if has_roc:
            print("ROC_AUC included.")

    return has_roc, roc_auc


def logreg_calibration_reference(y_type, scoring, models_params):
    lr = None
    lr_params = None

    if 'LogRClf_2nd' in models_params:
        lr = models_params['LogRClf_2nd'][0]
        if y_type == 'binary':
            lr.set_params(solver='liblinear')
        else:
            # y_type == 'multiclass'
            if scoring == 'neg_log_loss':
                lr.set_params(
                    solver='lbfgs', penalty='l2', multi_class='multinomial')
        lr_params = models_params['LogRClf_2nd'][1]
    else:
        if y_type == 'binary':
            lr =\
            pgd.full_search_models_and_parameters['LogRClf_2nd'][0].set_params(
                solver='liblinear')
            lr_params = pgd.full_search_models_and_parameters['LogRClf_2nd'][1]
        else:
            # y_type == 'multiclass'
            if scoring == 'neg_log_loss':
                # solver='saga', penalty='l1'
                lr =\
                pgd.full_search_models_and_parameters['LogRClf_2nd'][0].set_params(
                    solver='lbfgs', penalty='l2', multi_class='multinomial')
                lr_params = pgd.full_search_models_and_parameters['LogRClf_2nd'][1]

    return lr, lr_params


def classic_cv_calibration_process(
    ref_pred_score, training_estimator, final_estimator, X, y, 
    X_train, y_train, X_test, y_test, y_type, best_model_name, 
    best_model_estim, weights_test, weights_all, scoring, best_score, 
    best_nn_build_fn, nb_epoch, batch_size, tuning_method, models_data, 
    labels, d_name, serial):

    temp_estimator = training_estimator

    # check predicted probabilities for prediction confidence
    uncalibrated_data = eu.probability_confidence_before_calibration(
        temp_estimator, X_train, y_train, X_test, y_test, tuning_method,
        models_data, labels, serial
        )

    input_dim = 0

    if best_model_name in (
        'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
        'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'larger_deep_nn_Clf_2nd', 
        'deeper_nn_Clf_2nd'):
        del best_model_estim 
        input_dim = int(X_train.shape[1])
    else:
        del temp_estimator

    unc_pred_score = uncalibrated_data[0]
    unc_pipeline = uncalibrated_data[1]
    
    has_unc_roc, unc_roc_auc = check_model_hasroc("target", uncalibrated_data)

    predicted = unc_pipeline.predict(X_test)

    w_unc_acc = unc_pipeline.score(X_test, y_test, sample_weight=weights_test)*100

    confusion_matrix_and_clf_report(y_type, best_model_name, y_test, predicted)    
    print()

    # in case of LogRegression,
    # you should compare its probability curve against the ideal one

    # all stuff before plot of calibration curves should go into a function

    if unc_pred_score < ref_pred_score:
        print("'%s' is already well calibrated." % best_model_name)
        print("Let's resume metrics on test data.")

        if best_model_name not in (
            'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
            'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'larger_deep_nn_Clf_2nd', 
            'deeper_nn_Clf_2nd'):

            print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f'
                % (scoring.strip('neg_'), best_model_name, best_score))
            if has_unc_roc:
                print('Scoring [%s] of best uncalibrated ("%s") on test data: %1.3f'
                        % (scoring.strip('neg_'), best_model_name, unc_roc_auc))
            print('Accuracy of best uncalibrated ("%s") on test data: %.2f%%'
                    % (best_model_name, w_unc_acc))
            print()

            print("=== [task] Refit '%s' on all data." % best_model_name)
            print()
            print("X shape: ", X.shape)
            print("y shape: ", y.shape)
            print()

            # best_rscv_pipeline -> final_best_rscv_estimator
            eu.model_finalizer(
                final_estimator, X, y, scoring, tuning_method, d_name, serial)

    else:
        print("'%s' needs probability calibration." % best_model_name)

        if best_model_name in (
            'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
            'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'larger_deep_nn_Clf_2nd', 
            'deeper_nn_Clf_2nd'):

            print("Calibration of Keras neural networks is not implemented yet")
            print("""
            We assume this is the best calibration we can achieve
            with a Keras neural network, which approaches
            LogisticRegression's prediction confidence as
            the nr of iterations increases.
            """)

        else:

            temp_estimator = training_estimator

            # In case model needs calibration
            calib_data = eu.calibrate_probabilities(
                temp_estimator, X_train, y_train, X_test, y_test, 'sigmoid',
                tuning_method, models_data, kfold, labels, serial
                )

            calib_pred_score = calib_data[0]
            calib_pipeline = calib_data[1]

            has_calib_roc, calib_roc_auc = check_model_hasroc("target calib", calib_data)
            print()

            if calib_pred_score >= unc_pred_score:
                print("Sorry, we could not calibrate '%s' any better."
                    % best_model_name)
                print("We're rejecting calibrated '%s' and saving the uncalibrated one."
                    % best_model_name)

                print("Let's resume scores on validation and test data.")
                print('Mean cross-validated score [%s] of best uncalibrated ("%s"): %1.3f'
                    % (scoring.strip('neg_'), best_model_name, best_score))
                if has_unc_roc:
                    print('Scoring [%s] of best uncalibrated ("%s") on test data: %1.3f'
                        % (scoring.strip('neg_'), best_model_name, unc_roc_auc))
                print('Accuracy of best uncalibrated ("%s") on test data: %.2f%%'
                    % (best_model_name, w_unc_acc))
                print()

                # final_best_rscv_estimator
                print("=== [task] Refit '%s' on all data." % best_model_name)
                print()
                print("X shape: ", X.shape)
                print("y shape: ", y.shape)
                print()

                # best_rscv_pipeline -> final_best_rscv_estimator
                eu.model_finalizer(
                    final_estimator, X, y, scoring, tuning_method, d_name, serial)

            else:
                print("Achieved better calibration of model '%s'."
                    % best_model_name)
                print("Let's resume scores on test data.")
                print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f'
                    % (scoring.strip('neg_'), best_model_name, best_score))

                predicted = calib_pipeline.predict(X_test)

                print()
                print("After probability calibration...")

                confusion_matrix_and_clf_report(y_type, best_model_name, y_test, predicted) 
                print()

                w_calib_acc = calib_pipeline.score(
                    X_test, y_test, sample_weight=weights_test)*100

                if has_calib_roc:
                    print('Scoring [%s] of best calibrated ("%s") on test data: %1.3f'
                        % (scoring.strip('neg_'), best_model_name, calib_roc_auc))
                print('Accuracy of best calibrated ("%s") on test data: %.2f%%'
                    % (best_model_name, w_calib_acc))
                print()

                print("=== [task]: Train and calibrate probabilities of "
                    "pre-optimized model '%s' on all data."
                    % best_model_name)
                print()

                final_calib_pipeline = CalibratedClassifierCV(
                    final_estimator, method='sigmoid', cv=kfold)
                final_calib_pipeline.fit(X, y)

                fin_w_acc = final_calib_pipeline.score(X, y, sample_weight=weights_all)*100
                print('Overall accuracy of finalized best CCCV ("%s_rscv"): %.2f%%'
                    % (best_model_name, fin_w_acc))

                au.save_model(
                    final_calib_pipeline, best_model_name + '_final_calib_'
                    + tuning_method + '_' + serial + '.pkl', d_name=d_name
                    )

                # Uncomment to see pipeline, steps and params
                # print("Finalized calibrated best model '%s'." % best_model_name)
                # params = final_calib_pipeline.get_params()
                # for param_name in sorted(params.keys()):
                #     print("\t%s: %r" % (param_name, params[param_name]))
                # print()

    if best_model_name in (
        'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
        'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'larger_deep_nn_Clf_2nd', 
        'deeper_nn_Clf_2nd'):

        print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f' % (
                scoring, best_model_name, best_score))
        if has_unc_roc:
            print('Scoring [%s] of best ("%s") on test data: %1.3f' % (
                scoring.strip('neg_'), best_model_name, unc_roc_auc))
        print('Accuracy of best ("%s") on test data: %.2f%%'
                % (best_model_name, w_unc_acc))
        print()

        NN_transformer = final_estimator.fit(X, y)
        X_transformed = NN_transformer.transform(X)

        input_dim_final = int(X_transformed.shape[1])

        print()
        print("Input dimensions -- training: %d, finalization %d"
                % (input_dim, input_dim_final))
        print()

        f_name = best_model_name + '_feateng_for_keras_model_' + serial
        au.save_model(final_estimator, f_name + '.pkl', d_name=d_name)

        del final_estimator

        best_model_estim = best_keras_clf_estimator(
            y_type, best_nn_build_fn, nb_epoch, input_dim_final, labels, batch_size)

        steps_fin = []
        steps_fin.append((best_model_name, best_model_estim))

        untrained_NN_pipeline = Pipeline(steps_fin)

        eu.model_finalizer(
            untrained_NN_pipeline, X_transformed, y, scoring,
            tuning_method, d_name, serial
            )

        print()


def nested_cv_calibration_process(
    ref_pred_score, training_estimator, final_estimator, X, y, 
    X_train, y_train, X_test, y_test, y_type, best_model_name, 
    best_model_estim, weights_test, weights_all, scoring, best_score, 
    n_splits, n_iter, best_nn_build_fn, nb_epoch, param_grid, tuning_method, 
    models_data, labels, d_name, serial, random_state=0):

    temp_estimator = training_estimator

    print()

    # check predicted probabilities for prediction confidence
    uncalibrated_data = eu.probability_confidence_before_calibration(
        temp_estimator, X_train, y_train, X_test, y_test, tuning_method,
        models_data, labels, serial
        )

    # Keras stuff

    input_dim = 0
    if best_model_name == "KerasClf_2nd":
        input_dim = int(X_train.shape[1])
        del best_model_estim
    else:
        del temp_estimator

    unc_pred_score = uncalibrated_data[0]
    unc_pipeline = uncalibrated_data[1]
    
    has_unc_roc, unc_roc_auc = check_model_hasroc("target", uncalibrated_data)

    predicted = unc_pipeline.predict(X_test)

    w_unc_acc = unc_pipeline.score(X_test, y_test, sample_weight=weights_test)*100

    confusion_matrix_and_clf_report(y_type, best_model_name, y_test, predicted)    
    print()

    # in case of LogRegression,
    # you should compare its probability curve against the ideal one

    if unc_pred_score < ref_pred_score:
        print("'%s' is already well calibrated." % best_model_name)
        print("Let's resume metrics on test data.")

        if best_model_name != "KerasClf_2nd":

            print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f'
                % (scoring.strip('neg_'), best_model_name, best_score))
            if has_unc_roc:
                print('Scoring [%s] of best uncalibrated ("%s") on test data: %1.3f'
                        % (scoring.strip('neg_'), best_model_name, unc_roc_auc))
            print('Accuracy of best uncalibrated ("%s") on test data: %.2f%%'
                    % (best_model_name, w_unc_acc))
            print()

            print("=== [task] Refit '%s' on all data." % best_model_name)
            print()
            print("X shape: ", X.shape)
            print("y shape: ", y.shape)
            print()

            # best_rscv_pipeline -> final_best_rscv_estimator
            final_best_pipeline = eu.rscv_tuner(
                final_estimator, X, y, n_splits, param_grid, n_iter, scoring, 
                refit=True, cv_meth=tuning_method, random_state=random_state)

            print()
            print("Best estimator [%s]'s' params after hyp-tuning on all data."
                    % best_model_name)
            params = final_best_pipeline.get_params()
            for param_name in sorted(params.keys()):
                    print("\t%s: %r" % (param_name, params[param_name]))
            print()

            # input("Press key to continue... \n")

            w_best_acc = final_best_pipeline.score(
                X, y, sample_weight=weights_all)*100

            au.save_model(
                final_best_pipeline, best_model_name
                + '_final_nocalib_' + tuning_method + '_' + serial + '.pkl',
                d_name=d_name
                )

            print()

            # Uncomment to see pipeline, steps and params
            # print("Finalized uncalibrated best model '%s'."
            #       % best_model_name)
            # for step in final_best_rscv_pipeline.steps:
            #     print(type(step))
            #     print("step:", step[0])
            #     params = step[1].get_params()
            #     for param_name in sorted(params.keys()):
            #         print("\t%s: %r" % (param_name, params[param_name]))

    else:
        print("'%s' needs probability calibration." % best_model_name)
        
        if best_model_name == "KerasClf_2nd":

            print("Calibration of Keras neural networks is not implemented yet")
            print("""
            We assume this is the best calibration we can achieve
            with a Keras neural network, which approaches
            LogisticRegression's prediction confidence as
            the nr of iterations increases.
            """)
        
        else:

            temp_pipeline = training_estimator

            # In case model needs calibration
            calib_data = eu.calibrate_probabilities(
                temp_pipeline, X_train, y_train, X_test, y_test, 'sigmoid',
                tuning_method, models_data, kfold, labels, serial
                )

            calib_rscv_pred_score = calib_data[0]
            calib_rscv_pipeline = calib_data[1]
            
            has_calib_roc, calib_rscv_roc_auc = check_model_hasroc("target", calib_data)

            if calib_rscv_pred_score >= unc_pred_score:
                print("Sorry, we could not calibrate '%s' any better."
                        % best_model_name)
                print("Rejecting calibrated '%s' and saving the uncalibrated one."
                        % best_model_name)

                print("Let's resume metrics on test data.")
                print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f' % (
                    scoring.strip('neg_'), best_model_name, best_score))
                if has_unc_roc:
                    print('Scoring [%s] of best uncalibrated ("%s") on test data: %1.3f'
                            % (scoring.strip('neg_'), best_model_name, unc_roc_auc))
                print('Accuracy of best uncalibrated ("%s") on test data: %.2f%%'
                        % (best_model_name, w_unc_acc))
                print()

                # final_best_rscv_estimator
                final_best_rscv_pipeline = eu.rscv_tuner(
                    final_pipeline, X, y, n_splits, param_grid, n_iter,
                    scoring, refit=True, cv_meth=tuning_method, random_state=random_state)

                w_best_rscv_acc = final_best_rscv_pipeline.score(
                    X, y, sample_weight=w_all)*100

                print('Accuracy of best  ("%s") on all data: %.2f%%'
                        % (best_model_name, w_best_rscv_acc))
                print()

                au.save_model(
                    final_best_rscv_pipeline, best_model_name
                    + '_final_nocalib_' + tuning_method + '_'
                    + serial + '.pkl', d_name=d_name)

                print()

                # Uncomment to see pipeline, steps and params
                # print("Finalized uncalibrated best model '%s'." % best_model_name)
                # for step in final_best_rscv_pipeline.steps:
                #     print(type(step))
                #     print("step:", step[0])
                #     params = step[1].get_params()
                #     for param_name in sorted(params.keys()):
                #         print("\t%s: %r" % (param_name, params[param_name]))
                # print()

            else:
                print("Achieved better calibration of model '%s'."
                        % best_model_name)
                print("Let's resume scors on validation and test data.")
                print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f'
                        % (scoring.strip('neg_'), best_model_name, best_score))

                w_calib_rscv_acc = calib_rscv_pipeline.score(
                    X_test, y_test, sample_weight=w)*100

                if has_calib_roc:
                    print('Scoring [%s] of best uncalibrated ("%s") on test data: %1.3f%%'
                            % (scoring.strip('neg_'), best_model_name, calib_rscv_roc_auc))
                print('Accuracy of best calibrated ("%s") on test data: %.2f%%'
                        % (best_model_name, w_calib_rscv_acc))
                print()

                print("=== [task]: Tune '%s'' params with '%s' on all "
                        "data and calibrate probabilities."
                        % (best_model_name, tuning_method))
                print()

                best_rscv_parameters = eu.rscv_tuner(
                    final_pipeline, X, y, n_splits, param_grid, n_iter, scoring, 
                    refit=False, cv_meth=tuning_method, random_state=random_state
                    )

                temp_pipeline.set_params(**best_rscv_parameters)
                # calib_rscv_pipeline.named_steps[name].set_params(**best_parameters)

                final_calib_rscv_clf = CalibratedClassifierCV(
                    temp_pipeline, method='sigmoid', cv=kfold)
                final_calib_rscv_clf.fit(X, y)

                fin_w_rscv_acc = final_calib_rscv_clf.score(
                    X, y, sample_weight=w_all)*100
                print('Overall accuracy of finalized best CCCV ("%s_%s"): %.2f%%'
                        % (best_model_name, tuning_method, fin_w_rscv_acc))

                au.save_model(
                    final_calib_rscv_clf, best_model_name + '_final_calib_'
                    + tuning_method + '_' + serial + '.pkl', d_name=d_name)

    if best_model_name == "KerasClf_2nd":

        print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f' % (
                    scoring, best_model_name, best_score))
        if has_unc_roc:
            print('Scoring [%s] of best ("%s") on test data: %1.3f' % (
                scoring.strip('neg_'), best_model_name, unc_roc_auc))
        print('Accuracy of best ("%s") on test data: %.2f%%'
                % (best_model_name, w_unc_acc))
        print()
        
        NN_transformer = final_estimator.fit(X, y)
        X_transformed = NN_transformer.transform(X)

        input_dim_final = int(X_transformed.shape[1])

        print()
        print("Input dimensions -- training: %d, finalization %d"
                % (input_dim, input_dim_final))
        print()

        f_name = best_model_name + '_feateng_for_keras_model_' + serial

        au.save_model(final_estimator, f_name + '.pkl', d_name=d_name)

        # del final_pipeline

        if input_dim_final != input_dim:
            for n in np.arange(0, 3):
                param_grid[best_model_name + '__units_' + str(n)] = sp_randint(
                    input_dim_final, 5*input_dim_final)

        best_model_estim = best_keras_clf_estimator(
            y_type, best_nn_build_fn, nb_epoch, input_dim, labels)

        # finalize Keras clf
        steps_fin = [] 
        steps_fin.append((best_model_name, best_model_estim))

        untrained_NN_pipeline = Pipeline(steps_fin)

        # # final_best_rscv_pipeline
        # final_best_NN_pipeline = eu.rscv_tuner(
        #     untrained_NN_pipeline, X_transformed, y, n_splits, param_grid, n_iter,
        #     scoring, refit=True, random_state=random_state
        #     )
            
        # f_name = best_model_name + '_' + tuning_method + '_' + serial
        
        # keras_f_name = au.create_keras_model_filename(f_name, d_name=d_name)
        # final_best_NN_pipeline.named_steps[best_model_name].model.save(
        #     keras_f_name + '.h5')

        # this is equivalent to the process above
        final_best_NN_pipeline = eu.tune_and_evaluate(
            untrained_NN_pipeline, X_transformed, y, None, None, n_splits,
            param_grid, n_iter, scoring, [], refit=True,
            random_state=random_state, serial=serial, d_name=d_name, 
            save=True, cv_meth=tuning_method)

        w_best_NN_acc = final_best_NN_pipeline.score(
            X_transformed, y, sample_weight=weights_all)*100

        print('Accuracy of best ("%s") on all data: %.2f%%' % (
            best_model_name, w_best_NN_acc))
        print()

        # Uncomment to see pipeline, steps and params
        # print()
        # print("Best estimator [%s]'s' params after hyp-tuning on all data."
        #         % best_model_name)
        # for step in final_best_NN_pipeline.steps:
        #     print(type(step))
        #     print("step:", step[0])
        #     params = step[1].get_params()
        #     for param_name in sorted(params.keys()):
        #         print("\t%s: %r" % (param_name, params[param_name]))


def calibrate_best_model(
        X, y, X_train, X_test, y_train, y_test, tt_index, preprocessing,
        scores_of_best_model, all_models_and_parameters, n_splits,
        nb_epoch, scoring, models_data, d_name, random_state):
    """
    # calibrate best model from cross validation without hyperparameter tuning.

    ---
    ...
    """
    # Here start best model's calibration process

    best_model_name = scores_of_best_model[4][0]
    best_model_estim = scores_of_best_model[4][1]
    if best_model_name in (
        'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
            'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'larger_deep_nn_Clf_2nd', 
            'deeper_nn_Clf_2nd'):
        best_nn_build_fn = scores_of_best_model[4][2]
    else:
        best_nn_build_fn = None
    best_score = scores_of_best_model[0]
    best_score_dev = scores_of_best_model[1]
    best_cv_results = scores_of_best_model[2]
    # best_brier_score = scores_of_best_model[3]
    best_exec_time = scores_of_best_model[4]

    print()
    print("# Check prediction confidence of '%s' and eventually calibrate it."
          % best_model_name)
    print()

    # you should automatically know which encoding method to use: le or ohe

    encoding, scaler_tuple, featselector = preprocessing

    labels = None

    if 'labels' in all_models_and_parameters:
        labels = all_models_and_parameters['labels']
        print("Checking prediction confidence of multiclass '%s'"
              % best_model_name)
    else:
        print("No list of labels here. It's a binary problem.")

    # training pipeline

    steps = []
    # here you should also insert imputing and label encoding
    steps.append((best_model_name, best_model_estim))

    training_pipeline = Pipeline(steps)

    Y_type = mc.type_of_target(y)

    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    print("X sample:\n", X[:3])
    print("Y sample:\n", y[:3])
    print()

    # finalization pipeline -- for all models less Keras ones

    steps_fin = []
    # ...
    steps_fin.append(scaler_tuple)
    steps_fin.append(featselector)
    if best_model_name not in (
        'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
            'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'larger_deep_nn_Clf_2nd', 
            'deeper_nn_Clf_2nd'):
        steps_fin.append((best_model_name, best_model_estim))
    final_pipeline = Pipeline(steps_fin)

    # define serial nr. once and for all
    serial = "%04d" % randint(0, 1000)

    # Now, this model/pipeline might need calibration

    print()
    print("======= Training best estimator [%s], checking predicted "
          "probabilities and calibrating them" % best_model_name)

    # Train LogisticRegression for comparison of predicted probas

    kfold = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)

    w = eu.calculate_sample_weight(y_test)

    w_all = eu.calculate_sample_weight(y)

    # LogisticRegression as a calibration reference

    steps = []
    steps.append(('LogRClf_2nd', 
                  pgd.starting_point_models_and_params['LogRClf_2nd']))
    general_lr_pipeline = Pipeline(steps)

    temp_pipeline = general_lr_pipeline

    tuning_method = 'light_opt'

    print()
    # check predicted probabilities for prediction confidence
    uncalibrated_lr_data = eu.probability_confidence_before_calibration(
        temp_pipeline, X_train, y_train, X_test, y_test, tuning_method,
        models_data, labels, serial
        )

    del temp_pipeline

    lr_pred_score = uncalibrated_lr_data[0]
    lr_pipeline = uncalibrated_lr_data[1]

    has_roc, lr_roc_auc = check_model_hasroc(
        "LogRegression reference", uncalibrated_lr_data)

    print()

    predicted = lr_pipeline.predict(X_test)

    confusion_matrix_and_clf_report(Y_type, 'LogRClf_2nd', y_test, predicted)  
    print()
    print()

    # Evalute prediction confidence and, in case, calibrate

    # X = X.astype(np.float64)

    try:
        models_data[0]
    except IndexError:
        print("No LogRClf_2nd's data here. List 'models_data' is empty.")
    except Exception as e:
        print(e)
    else:
        print("LogRClf_2nd's data appended to models_data list")
        print()

    if best_model_name != "LogRClf_2nd":

        # eventually calibrating any ther model != LogReg

        input_dim = 0
        batch_size = 0

        if best_model_name in (
            'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
            'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'larger_deep_nn_Clf_2nd', 
            'deeper_nn_Clf_2nd'):

            # no hyperparam tuning for now

            tuning_method = 'None'

            input_dim = int(X_train.shape[1])
            batch_size = 32

            best_model_estim = best_keras_clf_estimator(
                Y_type, best_nn_build_fn, nb_epoch, input_dim, labels, batch_size)

        temp_pipeline = training_pipeline    # best model's pipeline

        # plot a learning curve

        print()
        print()
        print("[task] === Plot a learning curve")

        y_lim = None
        if Y_type == "binary":
            # minimum and maximum yvalues plotted in learning curve plot
            y_lim = (0.5, 1.01)

        au.plot_learning_curve(
            temp_pipeline, X_train, y_train, ylim=y_lim, cv=n_splits,
            scoring=scoring, n_jobs=-2, serial=serial,
            tuning=tuning_method, d_name=d_name
            )

        # plt.show()

        del temp_pipeline

        # temp_pipeline = training_pipeline

        print()

        # tune, etc.

        print()
        print("===== Check need for calibration")

        # here you should be able to automatically assess
        # whether current model in pipeline
        # actually needs calibration or not

        # if no calibration is needed,
        # you could finalize if you're happy with default hyperparameters
        # you could also compare
        # model(default_parameters) vs model(tuned_parameters)

        ###

        print("Check '%s''s prediction confidence after CV and calibrate probabilities."
                % best_model_name)
        print()

        # calibration returns models_data ...

        classic_cv_calibration_process(
            lr_pred_score, training_pipeline, final_pipeline, X, y, X_train, y_train, 
            X_test, y_test, Y_type, best_model_name, best_model_estim, w, w_all, 
            scoring, best_score, best_nn_build_fn, nb_epoch, batch_size, tuning_method, 
            models_data, labels, d_name, serial)

        if Y_type == 'binary':
            eu.plot_calibration_curves(
                y_test, best_model_name + '_' + tuning_method,
                models_data, 1, d_name
                )

            plt.show()

    else:
        # best_model_name == 'LogRClf_2nd':

        print()
        print()
        print("[task] === plotting a learning curve")
        print("Data:", d_name)
        print()

        y_lim = None
        if Y_type == "binary":
            y_lim = (0.5, 1.01)

        au.plot_learning_curve(
            lr_pipeline, X_train, y_train, ylim=y_lim,
            cv=kfold, scoring=scoring, n_jobs=-2, serial=serial,
            tuning=tuning_method, d_name=d_name
            )

        # plt.show()

        del lr_pipeline

        lr_pipeline = uncalibrated_lr_data[1]

        print()
        print("'LogRClf_2nd' is already well calibrated for definition!")
        print()
        print("Mean cv score [%s]: %1.3f"
              % (scoring.strip('neg_'), best_score))

        if has_roc:
            best_lr_roc_auc = lr_roc_auc
            print("ROC_AUC score on left-out data: %1.3f." % best_lr_roc_auc)
            print("- The higher, the better.")

        # refit with RSCV
        eu.model_finalizer(
            final_pipeline, X, y, scoring, tuning_method, d_name, serial)
        # best_lr_pipeline = final_best_lr_pipeline

        if Y_type == 'binary':
            eu.plot_calibration_curves(
                y_test, best_model_name + '_' + tuning_method, 
                models_data, 1, d_name)

            plt.show()

    plt.close('all')

    print()
    print()


def tune_calibrate_best_model(
        X, y, X_train, X_test, y_train, y_test, tt_index, preprocessing,
        scores_of_best_model, all_models_and_parameters, n_splits, n_iter,
        nb_epoch, scoring, models_data, d_name, random_state):
    """
    First line.

    ----------------------------------------------------------------------------
    ...
    """
    # Here start best model's calibration process
    best_model_name = scores_of_best_model[4][0]
    best_model_estim = scores_of_best_model[4][1]
    if best_model_name == "KerasClf_2nd":
        best_nn_build_fn = scores_of_best_model[4][2]
    else:
        best_nn_build_fn = None
    best_score = scores_of_best_model[0]
    best_score_dev = scores_of_best_model[1]
    # best_brier_score = scores_of_best_model[2]
    best_exec_time = scores_of_best_model[3]

    # here you should automatically know which encoding method to use: le or ohe

    encoding, scaler_tuple, featselector = preprocessing

    labels = None

    if 'labels' in all_models_and_parameters:
        labels = all_models_and_parameters['labels']
        print("Checking prediction confidence of multiclass '%s'"
              % best_model_name)
    else:
        print("No list of labels here. It's a binary problem.")

    # training pipeline

    steps = []
    # here you should also insert imputing and label encoding
    # steps.append(scaler_tuple)
    # for now
    # steps.append(featselector)
    steps.append((best_model_name, best_model_estim))

    training_pipeline = Pipeline(steps)

    Y_type = mc.type_of_target(y)

    # finalization pipeline -- for all models less Keras ones

    steps_fin = []
    steps_fin.append(scaler_tuple)
    steps_fin.append(featselector)
    if best_model_name != "KerasClf_2nd":
        steps_fin.append((best_model_name, best_model_estim))
    final_pipeline = Pipeline(steps_fin)

    # define serial nr. once and for all
    serial = "%04d" % randint(0, 1000)

    # Now, this model/pipeline might need calibration

    print()
    print("======= Tuning best estimator [%s], checking predicted "
          "probabilities and calibrating them" % best_model_name)

    # Train LogisticRegression for comparison of predicted probas

    # select param grid associated to resulting best model

    param_grid = dict()

    # retrieve n_iter for rscv/bscv from param_grid

    kfold = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)

    w = eu.calculate_sample_weight(y_test)
    w_all = eu.calculate_sample_weight(y)

    # LogisticRegression as a calibration reference

    lr, lr_params = logreg_calibration_reference(
        Y_type, scoring, all_models_and_parameters)

    steps = []
    steps.append(('LogRClf_2nd', lr))
    general_lr_pipeline = Pipeline(steps)

    temp_pipeline = general_lr_pipeline

    print()

    llr_n_iter = n_iter

    best_LogRClf_parameters = eu.tune_and_evaluate(
        temp_pipeline, X_train, y_train, X_test, y_test, n_splits,
        lr_params, llr_n_iter, scoring, models_data, refit=False, 
        random_state=random_state)

    temp_pipeline.set_params(**best_LogRClf_parameters)

    print()
    # check predicted probabilities for prediction confidence
    uncalibrated_lr_data = eu.probability_confidence_before_calibration(
        temp_pipeline, X_train, y_train, X_test, y_test, 'rscv', models_data,
        labels, serial
        )

    del temp_pipeline

    print()

    lr_pred_score = uncalibrated_lr_data[0]
    lr_pipeline = uncalibrated_lr_data[1]

    has_roc, lr_roc_auc = check_model_hasroc(
        "LogRegression reference", uncalibrated_lr_data)

    predicted = lr_pipeline.predict(X_test)

    confusion_matrix_and_clf_report(Y_type, 'LogRClf_2nd', y_test, predicted) 
    print()
    print()

    try:
        models_data[0]
    except IndexError:
        print("No LogRClf_2nd's data here. List 'models_data' is empty.")
    except Exception as e:
        print(e)
    else:
        print("LogRClf_2nd's data appended to models_data list")
        print()

    # Evalute prediction confidence and, in case, calibrate

    # X = X.astype(np.float64)    

    tuning_method = None

    if 'xscv' in all_models_and_parameters:
        # 'bscv'
        tuning_method = all_models_and_parameters['xscv']
    else:
        tuning_method = 'rscv'

    if best_model_name != "LogRClf_2nd":

        # tuning and eventually calibrating any other model !=  LogReg

        if best_model_name == "KerasClf_2nd":

            input_dim = int(X_train.shape[1])

            param_grid = pgd.Keras_param_grid

            # Keras Clf not included in 'all_models_and_parameters' dict

            # param_grid[best_model_name + '__units'] = sp_randint(input_dim, 5*input_dim)
            for n in np.arange(0, 3):
                param_grid[best_model_name + '__units_' + str(n)] = sp_randint(
                    input_dim, 5*input_dim)

        else:

            param_grid = all_models_and_parameters[best_model_name][1]

            if best_model_name == 'Bagging_SVMClf_2nd':
                # 'Bagging_SVMClf_2nd' made of 10 estimators
                del param_grid['Bagging_SVMClf_2nd__n_estimators']

        # tune, etc.

        print()
        print("===== Randomized Search CV")

        # here you should be able to automatically assess whether
        # current model in pipeline actually needs calibration or not

        # if no calibration is needed,
        # you could finalize if you're happy with default hyperparameters
        # you could also compare
        # model(default_parameters) vs model(tuned_parameters)

        print()
        print("Best model's [%s] parameter grid for RSCV:\n"
                % best_model_name, param_grid)
        print()

        temp_pipeline = training_pipeline    # best_pipeline

        # check that the total space of params >= n_iter

        # best_estimator_2nd

        best_parameters = eu.tune_and_evaluate(
            temp_pipeline, X_train, y_train, X_test, y_test, n_splits,
            param_grid, n_iter, scoring, models_data, refit=False,
            random_state=random_state, cv_meth=tuning_method)

        temp_pipeline.set_params(**best_parameters)

        # plot a learning curve

        print()
        print()
        print("[task] === plotting a learning curve")

        y_lim = None
        if Y_type == "binary":
            y_lim = (0.5, 1.01)

        au.plot_learning_curve(
            temp_pipeline, X_train, y_train, ylim=y_lim, cv=kfold,
            scoring=scoring, n_jobs=-2, serial=serial,
            tuning=tuning_method, d_name=d_name
            )

        # plt.show()

        del temp_pipeline

        ### calib fct

        print("Check '%s''s prediction confidence after (%s) CV and "
                "calibrate probabilities."
                % (best_model_name, tuning_method))
        print()

        nested_cv_calibration_process(
            lr_pred_score , training_pipeline, final_pipeline, X, y, 
            X_train, y_train, X_test, y_test, Y_type, best_model_name, 
            best_model_estim, w, w_all, scoring, best_score, n_splits, n_iter,
            best_nn_build_fn, nb_epoch, param_grid, tuning_method, models_data, 
            labels, d_name, serial, random_state)

        if Y_type == "binary":
            eu.plot_calibration_curves(
                y_test, best_model_name + '_' + tuning_method,
                models_data, 1, d_name
                )

            plt.show()

        print()
        print()

    else:
        # best_model_name=='LogRClf_2nd':

        # plot a learning curve

        print()
        print()
        print("[task] === plotting a learning curve")

        y_lim = None
        if Y_type == "binary":
            y_lim = (0.5, 1.01)

        au.plot_learning_curve(
            lr_pipeline, X_train, y_train, ylim=y_lim, cv=kfold,
            scoring=scoring, n_jobs=-2, serial=serial,
            tuning=tuning_method, d_name=d_name
            )

        # plt.show()

        del lr_pipeline
        lr_pipeline = uncalibrated_lr_data[1]

        print()
        print()
        print("'%s' is already well calibrated for definition!"
              % best_model_name)
        print()

        print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f'
              % (scoring.strip('neg_'), best_model_name, best_score))

        # best_lr_pipeline = lr_pipeline

        if lr_roc_auc is not None:
            print('Scoring [%s] of best uncalibrated ("%s") on test data: %1.3f'
                  % (scoring, best_model_name, lr_roc_auc))

        w_lr_acc = lr_pipeline.score(X_test, y_test, sample_weight=w)*100
        print('Accuracy of best uncalibrated ("%s") on test data: %.2f%%'
              % (best_model_name, w_lr_acc))
        print()

        # refit with RSCV; param_grid = pgd.LogR_param_grid
        final_best_lr_pipeline = eu.rscv_tuner(
            final_pipeline, X, y, n_splits, 
            all_models_and_parameters['LogRClf_2nd'][1], n_iter,
            scoring, refit=True, random_state=random_state
            )

        au.save_model(
            final_best_lr_pipeline, best_model_name + '_final_calib_rscv_'
            + serial + '.pkl', d_name=d_name)
        print()

        print("Performance on all data.")

        # w_all = calculate_sample_weight(y)

        # best_lr_pipeline
        w_lr_acc = final_best_lr_pipeline.score(X, y, sample_weight=w_all)*100

        print('Accuracy of best  ("%s") on all data: %.2f%%'
              % (best_model_name, w_lr_acc))
        print()

        # Uncomment to see pipeline, steps and params
        # print("Finalized '%s'." % best_model_name)
        # for step in final_best_lr_pipeline.steps:
        #     print("step:", step[0])
        #     params = step[1].get_params()
        #     for param_name in sorted(params.keys()):
        #         print("\t%s: %r" % (param_name, params[param_name]))
        # print()

        if Y_type == 'binary':
            eu.plot_calibration_curves(
                y_test, best_model_name + '_rscv', models_data, 
                1, d_name)

            plt.show()

    plt.close('all')

    print()
    print()
