"""

This module allows you to tune and calibrate the best estimator.

This module allows you to tune and calibrate the best estimator
returned by nested cv evaluation, and to just calibrate
the best estimator returned by the non-nested cv evaluation.

"""

from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals.joblib.my_exceptions import JoblibValueError

import numpy as np
import re
from random import randint
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
warnings.filterwarnings("ignore", category=FutureWarning)


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
            'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'deeper_nn_Clf_2nd'):
        best_nn_build_fn = scores_of_best_model[4][2]
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

    classes = None

    if 'labels' in all_models_and_parameters:
        classes = all_models_and_parameters['labels']
        print("Checking prediction confidence of multiclass '%s'"
              % best_model_name)
        output_dim = len(classes)
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
            'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'deeper_nn_Clf_2nd'):
        steps_fin.append((best_model_name, best_model_estim))
    final_pipeline = Pipeline(steps_fin)

    # define serial nr. once and for all
    serial = "%04d" % randint(0, 1000)

    # Now, this model/pipeline might need calibration

    print()
    print("======= Training best estimator [%s], checking predicted "
          "probabilities and calibrating them" % best_model_name)

    # GaussianNB does not need any tuning
    # VotingClassification needs no calibration, its estimators might do

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
        models_data, classes, serial
        )

    del temp_pipeline

    lr_pred_score = uncalibrated_lr_data[0]
    lr_pipeline = uncalibrated_lr_data[1]

    has_roc = 0
    try:
        uncalibrated_lr_data[2]
    except IndexError:
        print("ROC_AUC score is not available.")
    except Exception as e:
        print(e)
    else:
        lr_roc_auc = uncalibrated_lr_data[2]
        has_roc = 1
    finally:
        print("""
        We have LogRegression reference data
        to compare prediction confidence of models.
        """)
        if has_roc:
            print("ROC_AUC included.")
    print()

    predicted = lr_pipeline.predict(X_test)

    # if np.bincount(y_train).size == 2:
    if Y_type == 'binary':
        tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
        print()
        print("Test errors for 'LogRClf_2nd'")
        print("\ttrue negatives: %d" % tn)
        print("\tfalse positives: %d" % fp)
        print("\tfalse negatives: %d" % fn)
        print("\ttrue positives: %d" % tp)
    else:
        print()
        print("Confusion matrix for 'LogRClf_2nd'.\n",
              confusion_matrix(y_test, predicted))
        print()
    print("Classification report for 'LogRClf_2nd'\n",
          classification_report(y_test, predicted))
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

        # eventually calibrating any ther model != GaussianNB and LogReg

        if best_model_name in (
           'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
           'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'deeper_nn_Clf_2nd'):

            # no hyperparam tuning for now

            tuning_method = 'None'

            input_dim = int(X_train.shape[1])

            batch_size = 32

            y_lim = None

            if Y_type == 'binary':
                best_model_estim = KerasClassifier(
                    build_fn=best_nn_build_fn, nb_epoch=nb_epoch,
                    input_dim=input_dim, batch_size=batch_size, verbose=0
                )

                # minimum and maximum yvalues plotted in learning curve plot
                y_lim = (0.5, 1.01)
            elif Y_type == 'multiclass':
                best_model_estim = KerasClassifier(
                    build_fn=best_nn_build_fn, nb_epoch=nb_epoch,
                    input_dim=input_dim, output_dim=output_dim,
                    batch_size=batch_size, verbose=0
                )

            train_NN_pipeline = training_pipeline

            # plot a learning curve

            print()
            print()
            print("[task] === Plotting a learning curve")

            l_curve = 0

            try:
                au.plot_learning_curve(
                    train_NN_pipeline, X_train, y_train,
                    ylim=y_lim, cv=kfold, scoring=scoring, n_jobs=-2,
                    serial=serial, tuning=tuning_method, d_name=d_name
                    )

                plt.show()

                l_curve = 1

                del train_NN_pipeline

                # train_NN_pipeline = Pipeline(steps)
                train_NN_pipeline = training_pipeline

                print()
            except JoblibValueError as jve:
                print("Not able to complete learning process...")
            except ValueError as ve:
                print(ve)
            except Exception as e:
                print(e)
            finally:
                if not l_curve:
                    print("Sorry. Learning Curve plotting failed.")
                    print()

            # check predicted probabilities for prediction confidence
            NN_data = eu.probability_confidence_before_calibration(
                train_NN_pipeline, X_train, y_train, X_test, y_test,
                tuning_method, models_data, classes, serial
                )

            NN_pred_score = NN_data[0]
            NN_pipeline = NN_data[1]

            has_NN_roc = 0
            try:
                NN_data[2]
            except IndexError:
                print("ROC_AUC score is not available.")
            except Exception as e:
                print(e)
            else:
                NN_roc_auc = NN_data[2]
                has_NN_roc = 1
            finally:
                print("We have target data to compare prediction confidence "
                      "of models.")
                if has_NN_roc:
                    print("ROC_AUC included.")

            w_NN_acc = NN_pipeline.score(X_test, y_test, sample_weight=w)*100

            predicted = NN_pipeline.predict(X_test)

            if Y_type == 'binary':
                tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
                print()
                # true negatives, false positives, false negatives and
                # true positives for best estimator
                print("Test errors for '%s'" % best_model_name)
                print("\ttrue negatives: %d" % tn)
                print("\tfalse positives: %d" % fp)
                print("\tfalse negatives: %d" % fn)
                print("\ttrue positives: %d" % tp)
            else:
                print()
                print("Confusion matrix for '%s'.\n" %
                      best_model_name, confusion_matrix(y_test, predicted))
                print()
            print("Classification report for '%s'\n" %
                  best_model_name, classification_report(y_test, predicted))
            print()

            del steps, train_NN_pipeline, best_model_estim

            print()

            if NN_pred_score <= lr_pred_score:
                print("'%s' is already well calibrated." % best_model_name)
                print("Let's resume metrics on test data.")
            else:
                print("'%s' needs calibration, but it's not implemented yet,"
                      % best_model_name)
                print("here for Keras neural networks")
                print("""
                We assume this is the best calibration we can achieve
                with a Keras neural network, which approaches
                LogisticRegression's prediction confidence as
                the nr of iterations increases.
                """)
            print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f'
                  % (scoring, best_model_name, best_score))

            if has_NN_roc:
                print('Scoring [%s] of best ("%s") on test data: %1.3f'
                      % (scoring.strip('neg_'), best_model_name, NN_roc_auc))
            print('Accuracy of best ("%s") on test data: %.2f%%'
                  % (best_model_name, w_NN_acc))
            print()

            NN_transformer = final_pipeline.fit(X, y)
            X_transformed = NN_transformer.transform(X)

            input_dim_final = int(X_transformed.shape[1])

            print()
            print("Input dimensions -- training: %d, finalization %d"
                  % (input_dim, input_dim_final))
            print()

            f_name = best_model_name + '_feateng_for_keras_model_' + serial
            au.save_model(final_pipeline, f_name + '.pkl', d_name=d_name)

            del final_pipeline

            if Y_type == 'binary':
                best_model_estim = KerasClassifier(
                    build_fn=best_nn_build_fn, nb_epoch=nb_epoch,
                    input_dim=input_dim_final, batch_size=batch_size, verbose=0
                )
            elif Y_type == 'multiclass':
                best_model_estim = KerasClassifier(
                    build_fn=best_nn_build_fn, nb_epoch=nb_epoch,
                    input_dim=input_dim_final, output_dim=output_dim,
                    batch_size=batch_size, verbose=0
                )

            steps_fin = []
            steps_fin.append((best_model_name, best_model_estim))

            untrained_NN_pipeline = Pipeline(steps_fin)

            eu.model_finalizer(
                untrained_NN_pipeline, X_transformed, y, scoring,
                tuning_method, d_name, serial
                )

            print()

            if Y_type == "binary":

                eu.plot_calibration_curves(
                    y_test, best_model_name + '_' + tuning_method,
                    models_data, 1, d_name
                    )

                plt.show()

        else:

            temp_pipeline = training_pipeline    # best model's pipeline

            # plot a learning curve

            print()
            print()
            print("[task] === Plot a learning curve")

            y_lim = None
            if Y_type == "binary":
                y_lim = (0.5, 1.01)

            l_curve = 0
            try:
                au.plot_learning_curve(
                    temp_pipeline, X_train, y_train, ylim=y_lim, cv=n_splits,
                    scoring=scoring, n_jobs=-2, serial=serial,
                    tuning=tuning_method, d_name=d_name
                    )

                plt.show()

                l_curve = 1

                del temp_pipeline

                temp_pipeline = training_pipeline

                print()
            except JoblibValueError as jve:
                print("Not able to complete learning process...")
            except ValueError as ve:
                print(ve)
            except Exception as e:
                print(e)
            finally:
                if not l_curve:
                    print("Sorry. Learning Curve plotting failed.")
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

            # check predicted probabilities for prediction confidence
            uncalibrated_data = eu.probability_confidence_before_calibration(
                temp_pipeline, X_train, y_train, X_test, y_test, tuning_method,
                models_data, classes, serial
                )

            del temp_pipeline

            unc_pred_score = uncalibrated_data[0]
            unc_pipeline = uncalibrated_data[1]

            has_unc_roc = 0
            try:
                uncalibrated_data[2]
            except IndexError:
                print("ROC_AUC score is not available.")
            except Exception as e:
                print(e)
            else:
                unc_roc_auc = uncalibrated_data[2]
                has_unc_roc = 1
            finally:
                print("We have target data to compare prediction confidence "
                      "of models.")
                if has_unc_roc:
                    print("ROC_AUC included.")

            predicted = unc_pipeline.predict(X_test)

            if Y_type == 'binary':
                tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
                print()
                print("Test errors for '%s'" % best_model_name)
                print("\ttrue negatives: %d" % tn)
                print("\tfalse positives: %d" % fp)
                print("\tfalse negatives: %d" % fn)
                print("\ttrue positives: %d" % tp)
            else:
                print()
                print("Confusion matrix for '%s'.\n"
                      % best_model_name, confusion_matrix(y_test, predicted))
                print()
            print("Classification report for '%s'\n"
                  % best_model_name, classification_report(y_test, predicted))
            print()

            # input("Enter key to continue... \n")
            print()

            # in case of LogRegression,
            # you should compare its probability curve against the ideal one

            if unc_pred_score < lr_pred_score:
                print("'%s' is already well calibrated." % best_model_name)
                print("Let's resume metrics on test data.")

                # w = calculate_sample_weight(y_test)

                w_unc_acc = unc_pipeline.score(X_test, y_test, sample_weight=w)*100

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
                    final_pipeline, X, y, scoring, tuning_method, d_name, serial)

            else:
                print("'%s' needs probability calibration." % best_model_name)
                print()

                temp_pipeline = training_pipeline

                # In case model needs calibration
                calib_data = eu.calibrate_probabilities(
                    temp_pipeline, X_train, y_train, X_test, y_test, 'sigmoid',
                    tuning_method, models_data, kfold, serial
                    )

                calib_pred_score = calib_data[0]
                calib_pipeline = calib_data[1]

                has_calib_roc = 0
                try:
                    calib_data[2]
                except IndexError:
                    print("ROC_AUC score is not available.")
                except Exception as e:
                    print(e)
                else:
                    calib_roc_auc = calib_data[2]
                    has_calib_roc = 1
                finally:
                    print("We have target calib data to compare prediction "
                          "confidence of models.")
                    if has_roc:
                        print("ROC_AUC included.")
                print()

                if calib_pred_score >= unc_pred_score:
                    print("Sorry, we could not calibrate '%s' any better."
                          % best_model_name)
                    print("We're rejecting calibrated '%s' and saving the uncalibrated one."
                          % best_model_name)

                    w_unc_acc = unc_pipeline.score(X_test, y_test, sample_weight=w)*100

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
                        final_pipeline, X, y, scoring, tuning_method, d_name, serial)

                else:
                    print("Achieved better calibration of model '%s'."
                          % best_model_name)
                    print("Let's resume scores on test data.")
                    print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f'
                          % (scoring.strip('neg_'), best_model_name, best_score))

                    predicted = calib_pipeline.predict(X_test)

                    print()
                    print("After probability calibration...")

                    if Y_type == 'binary':
                        tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
                        print()
                        print("Test errors for '%s'" % best_model_name)
                        print("\ttrue negatives: %d" % tn)
                        print("\tfalse positives: %d" % fp)
                        print("\tfalse negatives: %d" % fn)
                        print("\ttrue positives: %d" % tp)
                    else:
                        print()
                        print("Confusion matrix for calibrated '%s'.\n"
                              % best_model_name, confusion_matrix(y_test, predicted))
                        print()
                    print("Classification report for calibrated '%s'\n"
                          % best_model_name, classification_report(y_test, predicted))
                    print()

                    w_calib_acc = calib_pipeline.score(
                        X_test, y_test, sample_weight=w)*100

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
                        final_pipeline, method='sigmoid', cv=kfold)
                    final_calib_pipeline.fit(X, y)

                    fin_w_acc = final_calib_pipeline.score(X, y, sample_weight=w_all)*100
                    print('Overall accuracy of finalized best CCCV ("%s_rscv"): %.2f%%'
                          % (best_model_name, fin_w_acc))

                    au.save_model(
                        final_calib_pipeline, best_model_name + '_final_calib_'
                        + tuning_method + '_' + serial + '.pkl', d_name=d_name
                        )

                    # final
                    # Uncomment to see pipeline, steps and params
                    """
                    print("Finalized calibrated best model '%s'." % best_model_name)
                    params = final_calib_pipeline.get_params()
                    for param_name in sorted(params.keys()):
                        print("\t%s: %r" % (param_name, params[param_name]))
                    print()
                    """

            if Y_type == 'binary':
                eu.plot_calibration_curves(
                    y_test, best_model_name + '_' + tuning_method,
                    models_data, 1, d_name
                    )

                plt.show()
                # plt.close()

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

        l_curve = 0
        try:
            au.plot_learning_curve(
                lr_pipeline, X_train, y_train, ylim=y_lim,
                cv=kfold, scoring=scoring, n_jobs=-2, serial=serial,
                tuning=tuning_method, d_name=d_name
                )

            plt.show()

            l_curve = 1

            del lr_pipeline

            lr_pipeline = uncalibrated_lr_data[1]

            print()
        except JoblibValueError as jve:
            print("Not able to complete learning process...")
        except ValueError as ve:
            print(ve)
        except Exception as e:
            print(e)
        finally:
            if not l_curve:
                print("Sorry. Learning Curve plotting failed.")
                print()

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
    if best_model_name in (
        'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
            'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'deeper_nn_Clf_2nd'
            ):
        best_nn_build_fn = scores_of_best_model[4][2]
    best_score = scores_of_best_model[0]
    best_score_dev = scores_of_best_model[1]
    # best_brier_score = scores_of_best_model[2]
    best_exec_time = scores_of_best_model[3]

    # here you should automatically know which encoding method to use: le or ohe

    encoding, scaler_tuple, featselector = preprocessing

    classes = None

    if 'labels' in all_models_and_parameters:
        classes = all_models_and_parameters['labels']
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
    if best_model_name not in (
        'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
            'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'deeper_nn_Clf_2nd'):
        steps_fin.append((best_model_name, best_model_estim))
    final_pipeline = Pipeline(steps_fin)

    # define serial nr. once and for all
    serial = "%04d" % randint(0, 1000)

    # Now, this model/pipeline might need calibration

    print()
    print("======= Tuning best estimator [%s], checking predicted "
          "probabilities and calibrating them" % best_model_name)

    # GaussianNB does not need any tuning
    # VotingClassification needs no calibration, its estimators might do

    # Train LogisticRegression for comparison of predicted probas

    # select param grid associated to resulting best model

    param_grid = dict()

    kfold = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)

    w = eu.calculate_sample_weight(y_test)
    w_all = eu.calculate_sample_weight(y)

    # LogisticRegression as a calibration reference

    lr = None
    lr_params = None

    if 'LogRClf_2nd' in all_models_and_parameters:
        lr = all_models_and_parameters['LogRClf_2nd'][0]
        if Y_type == 'binary':
            lr.set_params(solver='liblinear')
        else:
            # Y_type == 'multiclass'
            if scoring == 'neg_log_loss':
                lr.set_params(
                    solver='lbfgs', penalty='l2', multi_class='multinomial')
        lr_params = all_models_and_parameters['LogRClf_2nd'][1]
    else:
        if Y_type == 'binary':
            lr =\
            pgd.full_search_models_and_parameters['LogRClf_2nd'][0].set_params(
                solver='liblinear')
            lr_params = pgd.full_search_models_and_parameters['LogRClf_2nd'][1]
        else:
            # Y_type == 'multiclass'
            if scoring == 'neg_log_loss':
                # solver='saga', penalty='l1'
                lr =\
                pgd.full_search_models_and_parameters['LogRClf_2nd'][0].set_params(
                    solver='lbfgs', penalty='l2', multi_class='multinomial')
                lr_params = pgd.full_search_models_and_parameters['LogRClf_2nd'][1]

    steps = []
    steps.append(('LogRClf_2nd', 
                  # pgd.full_search_models_and_parameters['LogRClf_2nd'][0]
                  lr
                  ))
    general_lr_pipeline = Pipeline(steps)

    temp_pipeline = general_lr_pipeline

    print()

    llr_n_iter = n_iter

    best_LogRClf_parameters = eu.tune_and_evaluate(
        temp_pipeline, X_train, y_train, X_test, y_test, n_splits,
        lr_params, llr_n_iter, scoring, models_data, refit=False, 
        random_state=random_state
        )

    temp_pipeline.set_params(**best_LogRClf_parameters)

    print()
    # check predicted probabilities for prediction confidence
    uncalibrated_lr_data = eu.probability_confidence_before_calibration(
        temp_pipeline, X_train, y_train, X_test, y_test, 'rscv', models_data,
        classes, serial
        )

    del temp_pipeline

    print()

    lr_pred_score = uncalibrated_lr_data[0]
    lr_pipeline = uncalibrated_lr_data[1]

    has_roc = 0
    try:
        uncalibrated_lr_data[2]
    except IndexError:
        print("ROC_AUC score is not available.")
    except Exception as e:
        print(e)
    else:
        lr_roc_auc = uncalibrated_lr_data[2]
        has_roc = 1
    finally:
        print("We have LogRegression reference data to compare prediction "
              "confidence of models.")
        if has_roc:
            print("ROC_AUC included.")

    predicted = lr_pipeline.predict(X_test)

    # if np.bincount(y_train).size == 2:
    if Y_type == 'binary':
        print()
        tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
        print("Test errors for 'LogRClf_2nd'")
        print("\ttrue negatives: %d" % tn)
        print("\tfalse positives: %d" % fp)
        print("\tfalse negatives: %d" % fn)
        print("\ttrue positives: %d" % tp)
    else:
        print()
        print("Confusion matrix for 'LogRClf_2nd'.\n",
              confusion_matrix(y_test, predicted))
        print()
    print("Classification report for 'LogRClf_2nd'\n",
          classification_report(y_test, predicted))
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

    tuning_method = 'rscv'

    if best_model_name != "LogRClf_2nd":

        if best_model_name in (
            'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
                'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'deeper_nn_Clf_2nd'):

            # no hyperparam tuning for now

            tuning_method = 'None'

            input_dim = int(X_train.shape[1])

            batch_size = 32   # 5

            best_model_estim = KerasClassifier(
                build_fn=best_nn_build_fn, nb_epoch=nb_epoch,
                input_dim=input_dim, batch_size=batch_size, verbose=0
                )

            train_NN_pipeline = training_pipeline

            # plot a learning curve

            print()
            print()
            print("[task] === plotting a learning curve")

            y_lim = None
            if Y_type == "binary":
                y_lim = (0.5, 1.01)

            l_curve = 0
            try:
                au.plot_learning_curve(
                    train_NN_pipeline, X_train, y_train, ylim=y_lim, cv=kfold,
                    scoring=scoring, n_jobs=-2, serial=serial,
                    tuning=tuning_method, d_name=d_name
                    )

                plt.show()
            except JoblibValueError as jve:
                print("Not able to complete learning process...")
            except ValueError as ve:
                print(ve)
            except Exception as e:
                print(e)
            else:
                l_curve = 1

                del train_NN_pipeline

                train_NN_pipeline = training_pipeline

                print()
            finally:
                if not l_curve:
                    print("Sorry. Learning Curve plotting failed.")
                    print()

            # check predicted probabilities for prediction confidence
            NN_data = eu.probability_confidence_before_calibration(
                train_NN_pipeline, X_train, y_train, X_test, y_test,
                tuning_method, models_data, classes, serial
                )

            NN_pred_score = NN_data[0]
            NN_pipeline = NN_data[1]

            has_NN_roc = 0
            try:
                NN_data[2]
            except IndexError:
                print("ROC_AUC score is not available.")
            except Exception as e:
                print(e)
            else:
                NN_roc_auc = NN_data[2]
                has_NN_roc = 1
            finally:
                pass

            predicted = NN_pipeline.predict(X_test)

            if Y_type == 'binary':
                tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
                print()
                print("Test errors for '%s'" % best_model_name)
                print("\ttrue negatives: %d" % tn)
                print("\tfalse positives: %d" % fp)
                print("\tfalse negatives: %d" % fn)
                print("\ttrue positives: %d" % tp)
            else:
                print()
                print("Confusion matrix for '%s'.\n"
                      % best_model_name, confusion_matrix(y_test, predicted))
                print()
            print("Classification report for '%s'\n"
                  % best_model_name, classification_report(y_test, predicted))
            print()

            w_NN_acc = NN_pipeline.score(X_test, y_test, sample_weight=w)*100

            del steps, train_NN_pipeline, best_model_estim

            print()

            if NN_pred_score <= lr_pred_score:
                print("'%s' is already well calibrated." % best_model_name)
                print("Let's resume metrics on test data.")
            else:
                print("'%s' needs calibration, but it's not implemented yet,"
                      % best_model_name)
                print("here for Keras neural networks.")
                print("""
                We assume this is the best calibration we can achieve
                with a Keras neural network, which approaches
                LogisticRegression's prediction confidence as
                the nr of iterations increases.
                """)
            print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f'
                  % (scoring.strip('neg_'), best_model_name, best_score))

            if has_NN_roc:
                print('Scoring [%s] of best ("%s") on test data: %1.3f'
                      % (scoring.strip('neg_'), best_model_name, NN_roc_auc))
            print('Accuracy of best ("%s") on test data: %.2f%%'
                  % (best_model_name, w_NN_acc))
            print()

            NN_transformer = final_pipeline.fit(X, y)
            X_transformed = NN_transformer.transform(X)

            input_dim_final = int(X_transformed.shape[1])

            print()
            print("Input dimensions -- training: %d, finalization %d"
                  % (input_dim, input_dim_final))
            print()

            f_name = best_model_name + '_feateng_for_keras_model_' + serial

            au.save_model(final_pipeline, f_name + '.pkl', d_name=d_name)

            del final_pipeline

            best_model_estim = KerasClassifier(
                build_fn=best_nn_build_fn, nb_epoch=nb_epoch,
                input_dim=input_dim_final, batch_size=batch_size, verbose=0
                )

            steps_fin = []
            steps_fin.append((best_model_name, best_model_estim))

            untrained_NN_pipeline = Pipeline(steps_fin)

            eu.model_finalizer(
                untrained_NN_pipeline, X_transformed, y, scoring,
                tuning_method, d_name, serial
                )

            print()

            if Y_type == 'binary':

                eu.plot_calibration_curves(
                    y_test, best_model_name + '_' + tuning_method,
                    models_data, 1, d_name
                    )

                plt.show()

            print()

        elif best_model_name == 'GaussianNBClf_2nd':

            # no hyperparam tuning -- default: priors == None

            # You should refactor this

            tuning_method = 'None'

            pipeline = training_pipeline

            # plot a learning curve

            print()
            print()
            print("[task] === plotting a learning curve")

            y_lim = None
            if Y_type == "binary":
                y_lim = (0.5, 1.01)

            l_curve = 0
            try:
                au.plot_learning_curve(
                    pipeline, X_train, y_train, ylim=y_lim, cv=kfold,
                    scoring=scoring, n_jobs=-2, serial=serial,
                    tuning=tuning_method, d_name=d_name
                    )

                plt.show()
            except JoblibValueError as jve:
                print("Not able to complete learning process...")
            except ValueError as ve:
                print(ve)
            except Exception as e:
                print(e)
            else:
                l_curve = 1

                del pipeline

                pipeline = training_pipeline

                print()
            finally:
                if not l_curve:
                    print("Sorry. Learning Curve plotting failed.")
                    print()

            # check predicted probabilities for prediction confidence
            GNB_unc_data = eu.probability_confidence_before_calibration(
                pipeline, X_train, y_train, X_test, y_test, tuning_method,
                models_data, classes, serial
                )

            GNB_pred_score = GNB_unc_data[0]
            GNB_pipeline = GNB_unc_data[1]

            has_GNB_roc = 0
            try:
                GNB_unc_data[2]
            except IndexError:
                print("ROC_AUC score is not available.")
            except Exception as e:
                print(e)
            else:
                GNB_roc_auc = GNB_unc_data[2]
                has_GNB_roc = 1
            finally:
                print("We have LogRegression reference data to compare "
                      "prediction confidence of models.")
                if has_GNB_roc:
                    print("ROC_AUC included.")

            predicted = GNB_pipeline.predict(X_test)

            print()
            print("Before probability calibration...")

            if Y_type == 'binary':
                tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
                print()
                print("Test errors for '%s'" % best_model_name)
                print("\ttrue negatives: %d" % tn)
                print("\tfalse positives: %d" % fp)
                print("\tfalse negatives: %d" % fn)
                print("\ttrue positives: %d" % tp)
            else:
                print()
                print("Confusion matrix for '%s'.\n"
                      % best_model_name, confusion_matrix(y_test, predicted))
                print()
            print("Classification report for '%s'\n"
                  % best_model_name, classification_report(y_test, predicted))
            print()

            w_GNB_acc = GNB_pipeline.score(X_test, y_test, sample_weight=w)*100

            print()

            if GNB_pred_score <= lr_pred_score:
                print("'%s' is already well calibrated." % best_model_name)
                print("Let's resume metrics on test data.")
                print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f'
                      % (scoring.strip('neg_'), best_model_name, best_score))

                # w_GNB_acc = GNB_pipeline.score(X_test, y_test, sample_weight=w)*100
                if has_GNB_roc:
                    print('Scoring [%s] of best uncalibrated ("%s") on test data: %1.3f'
                          % (scoring.strip('neg_'), best_model_name, GNB_roc_auc))
                print('Accuracy of best uncalibrated ("%s") on test data: %.2f%%'
                      % (best_model_name, w_GNB_acc))
                print()

                # GNB_pipeline
                eu.model_finalizer(
                    final_pipeline, X, y, scoring, tuning_method, d_name, serial)
                print()

            else:
                print("'%s' needs probability calibration." % best_model_name)
                print()

                # In case model needed calibration
                calib_GNB_data = eu.calibrate_probabilities(
                    pipeline, X_train, y_train, X_test, y_test, 'sigmoid',
                    tuning_method, models_data, kfold, serial
                    )
                print()

                calib_GNB_pred_score = calib_GNB_data[0]
                calib_GNB_pipeline = calib_GNB_data[1]

                has_calib_GNB_roc = 0
                try:
                    calib_GNB_data[2]
                except IndexError:
                    print("ROC_AUC score is not available.")
                except Exception as e:
                    print(e)
                else:
                    calib_GNB_roc_auc = calib_GNB_data[2]
                    has_GNB_roc = 1
                finally:
                    if has_calib_GNB_roc:
                        print("ROC_AUC included.")

                if calib_GNB_pred_score >= GNB_pred_score:
                    print("Sorry, we could not calibrate '%s' any better."
                          % best_model_name)
                    print("Rejecting calibrated '%s' and saving the uncalibrated one."
                          % best_model_name)

                    print("Let's resume scores on validation and test data.")
                    print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f'
                          % (scoring.strip('neg_'), best_model_name, best_score))
                    if has_GNB_roc:
                        print('Scoring [%s] of best uncalibrated ("%s") on test data: %1.3f'
                              % (scoring.strip('neg_'), best_model_name, GNB_roc_auc))
                    print('Accuracy of best uncalibrated ("%s") on test data: %.2f%%'
                          % (best_model_name, w_GNB_acc))
                    print()

                    eu.model_finalizer(
                        final_pipeline, X, y, scoring, tuning_method, d_name, serial)
                    print()

                else:
                    print("Achieved better calibration of model '%s'." % best_model_name)
                    print("Let's resume scores on validation and test data.")
                    print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f'
                          % (scoring.strip(), best_model_name, best_score))

                    predicted = calib_GNB_pipeline.predict(X_test)

                    print()
                    print("After probability calibration...")

                    if Y_type == 'binary':
                        tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
                        print()
                        print("Test errors for '%s'" % best_model_name)
                        print("\ttrue negatives: %d" % tn)
                        print("\tfalse positives: %d" % fp)
                        print("\tfalse negatives: %d" % fn)
                        print("\ttrue positives: %d" % tp)
                    else:
                        print()
                        print("Confusion matrix for calibrated '%s'.\n"
                              % best_model_name, confusion_matrix(y_test, predicted))
                        print()
                    print("Classification report for calibrated '%s'\n"
                          % best_model_name, classification_report(y_test, predicted))
                    print()

                    w_GNB_acc = calib_GNB_pipeline.score(X_test, y_test, sample_weight=w)*100

                    if has_calib_GNB_roc:
                        print('Scoring [%s] of best calibrated ("%s") on test data: %1.3f'
                              % (scoring.strip(), best_model_name, calib_GNB_roc_auc))
                    print('Accuracy of best calibrated ("%s") on test data: %.2f%%'
                          % (best_model_name, w_GNB_acc))
                    print()
                    print("=== [task]: Train and calibrate '%s on all data")

                    # finalize model - tune hyperparameters on all data, calibrate
                    # GaussianNB needs no tuning, just calibrate on all data

                    final_calib_GNB_clf = CalibratedClassifierCV(
                        final_pipeline, method='sigmoid', cv=kfold)
                    final_calib_GNB_clf.fit(X, y)

                    fin_w_GNB_acc = final_calib_GNB_clf.score(X, y, sample_weight=w_all)*100
                    print("Overall accuracy of finalized best CCCV ('%s'): %.2f%%"
                          % (best_model_name, fin_w_GNB_acc))

                    f_name = best_model_name + '_final_calib_' + tuning_method + '_' + serial

                    au.save_model(final_calib_GNB_clf, f_name + '.pkl', d_name=d_name)

            if Y_type == 'binary':

                eu.plot_calibration_curves(
                    y_test, best_model_name + '_' + tuning_method, 
                    models_data, 1, d_name
                    )

                plt.show()

            print()

        elif best_model_name == 'VClf_3_2nd':

            print("======= RSCV of VotingClassifier w top 3 models")
            print()
            print("=== [task] Tune and eventually calibrate top 3 models")
            print()
            print()

            # calibrate top 3 models if needed

            # Tuning hyperparameters separately, first tune each model

            top_3_models_data = all_models_and_parameters['VClf_3_2nd'][0]

            print("Top 3 models data:\n", top_3_models_data)

            # input("Press key to continue...")
            print()

            train_idx, test_idx = tt_index

            if Y_type == 'binary':

                # X_train, X_valid, y_train, y_valid = train_test_split(
                #     X_train, y_train, stratify=y_train, test_size=0.2,
                #     random_state=random_state
                #     )

                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=0.2, random_state=random_state)

                train_idx, val_idx = (None, None)

                for trn, val in sss.split(X_train, y_train):
                    train_idx, val_idx = trn, val

                X_train, X_valid = X_train[train_idx], X_train[val_idx]
                y_train, y_valid = y_train.iloc[train_idx], y_train.iloc[val_idx]

                vc3_estimators = eu.calibrate_estimators_for_soft_voting(
                    top_3_models_data, X_train, y_train, X_valid, y_valid,
                    X_test, y_test, n_splits, n_iter, scoring, lr_pred_score,
                    kfold, tuning_method
                    )

                refit_msg = " less 'valid'"

            else:

                vc3_estimators = []

                nr_clf = 1
                for k, v in top_3_models_data.items():
                    print("Estimator nr. %d: %s" % (nr_clf, k))
                    p = Pipeline([(k, v[3])])
                    vc3_estimators.append(('p' + str(nr_clf), p))
                    nr_clf += 1

                # print("VClf3 estimators -- top 3 models':", vc3_estimators)
                # print()

                list_of_params = [v[4] for k, v in top_3_models_data.items()]

                print("List of VClf3 params:", list_of_params)
                print()
                # input("Press any key to continue...")
                # print()

                vc3_params = dict()

                nr_clf = 1
                for par in list_of_params:
                    new_par = {
                        'VClf_3_2nd__p' + str(nr_clf) + '__'
                        + k: v for k, v in par.items()
                        }
                    vc3_params.update(new_par)
                    nr_clf += 1

                vc3_param_grid = {
                    'VClf_3_2nd__'
                    + k: v for k, v in pgd.VC_3_param_grid.items()
                    }

                vc3_params.update(vc3_param_grid)

                refit_msg = ''

            print("VClf3 estimators -- top 3 models':", vc3_estimators)
            print()

            # nested-cv of VotingClassifier

            # top 3 estimators now are well calibrated # rscv_vclf3
            vclf3 = VotingClassifier(vc3_estimators, voting='soft')

            # rscv on VotingClassifier to select weights

            print()
            print("Steps:\n", steps)
            # input("Enter key to continue...")

            # deleting step w LogRClf_2nd

            training_pipeline.steps.pop(-1)

            training_pipeline.steps.append(('VClf_3_2nd', vclf3))

            # vclf3_pipeline = Pipeline(steps)

            vclf3_pipeline = training_pipeline

            temp_pipeline = vclf3_pipeline

            # vclf3_pipeline <-- vclf3

            # ...then tune VotingClassifier's weights

            vc3_params = {
                'VClf_3_2nd__' + k: v for k, v in pgd.VC_3_param_grid.items()
                }

            vc3_n_iter = au.check_search_space_of_params(n_iter, vc3_params)

            best_vclf3_params = eu.tune_and_evaluate(
                temp_pipeline, X_train, y_train, X_test, y_test, n_splits,
                vc3_params, vc3_n_iter, scoring, models_data, refit=False,
                random_state=random_state
                )

            print()
            print("Parameters of completely tuned VotingClassifier:\n",
                  best_vclf3_params)
            print()

            temp_pipeline.set_params(**best_vclf3_params)

            tuned_vclf3_pipeline = temp_pipeline

            # plot a learning curve

            print()
            print()
            print("[task] === plotting a learning curve")

            y_lim = None
            if Y_type == "binary":
                y_lim = (0.5, 1.01)

            l_curve = 0
            try:
                au.plot_learning_curve(
                    tuned_vclf3_pipeline, X_train, y_train, ylim=y_lim,
                    cv=kfold, scoring=scoring, n_jobs=-2, serial=serial,
                    tuning=tuning_method, d_name=d_name
                    )

                plt.show()
            except JoblibValueError as jve:
                print("Not able to complete learning process...")
            except ValueError as ve:
                print(ve)
            except Exception as e:
                print(e)
            else:
                l_curve = 1

                del tuned_vclf3_pipeline
                tuned_vclf3_pipeline = temp_pipeline

                print()
            finally:
                if not l_curve:
                    print("Sorry. Learning Curve plotting failed.")
                    print()

            # this fct also fits model/pipeline
            vclf3_data = eu.probability_confidence_before_calibration(
                    tuned_vclf3_pipeline, X_train, y_train, X_test, y_test,
                    tuning_method, models_data, classes, serial
                    )

            # trained_vclf3_pred_score = vclf3_data[0]
            trained_vclf3_pipeline = vclf3_data[1]

            has_vc3_roc = 0
            try:
                vclf3_data[2]
            except IndexError:
                print("ROC_AUC score is not available.")
            except Exception as e:
                print(e)
            else:
                trained_vclf3_roc_auc = vclf3_data[2]
                has_vc3_roc = 1
            finally:
                print("We have target data to compare prediction confidence "
                      "of models.")
                if has_vc3_roc:
                    print("ROC_AUC included.")

            predicted = trained_vclf3_pipeline.predict(X_test)

            print()

            if Y_type == 'binary':
                tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
                print()
                print("Test errors for '%s'" % best_model_name)
                print("\ttrue negatives: %d" % tn)
                print("\tfalse positives: %d" % fp)
                print("\tfalse negatives: %d" % fn)
                print("\ttrue positives: %d" % tp)
            else:
                print()
                print("Confusion matrix for calibrated '%s'.\n"
                      % best_model_name, confusion_matrix(y_test, predicted))
                print()
            print("Classification report for calibrated '%s'\n"
                  % best_model_name, classification_report(y_test, predicted))
            print()

            w_vclf3_acc = trained_vclf3_pipeline.score(
                X_test, y_test, sample_weight=w)*100
            # predictions = tuned_vclf3_pipeline.predict(X_test)

            print()
            print("Let's resume scores on validation and test data.")
            print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f'
                  % (scoring.strip('neg_'), best_model_name, best_score))
            if has_vc3_roc:
                print("ROC_AUC of 'VClf_3_2nd' on test data: %1.3f"
                      % trained_vclf3_roc_auc)
            print("Accuracy of best 'VClf_3_2nd' on test data: %.2f%%"
                  % w_vclf3_acc)
            print()

            # input("Press key to continue...")
            print()

            print("=== [task] Refit 'VClf_3_2nd' on all data%s." % refit_msg)
            print()

            # calibrate top 3 models once again on all data...

            final_idx = np.concatenate((train_idx, test_idx), axis=0)
            X_final = X.iloc[final_idx]
            y_final = y.iloc[final_idx]

            print("X shape: ", X_final.shape)
            print("y shape: ", y_final.shape)
            print("X sample:\n", X_final[:3])
            print("Y sample:\n", y_final[:3])
            print()

            vclf3 = VotingClassifier(vc3_estimators, voting='soft')

            steps_fin.pop(-1)

            print()
            print("Steps:\n", steps_fin)
            # input("Enter key to continue...")

            steps_fin.append(('VClf_3_2nd', vclf3))

            vclf3_pipeline = Pipeline(steps_fin)

            print()
            print("VotingClassifier w calibrated estimators:", vclf3_pipeline)
            print()

            # vclf3_pipeline
            best_vclf3_estimator = eu.rscv_tuner(
                vclf3_pipeline, X_final, y_final, n_splits, vc3_params,
                vc3_n_iter, scoring, refit=True, random_state=random_state
                )

            # final

            f_name = 'VClf_3_2nd__no_calib_' + tuning_method + '_' + serial

            au.save_model(best_vclf3_estimator, f_name + '.pkl', d_name=d_name)

            # Uncomment to see pipeline, steps and params
            """
            print()
            print("Finalized best model '%s'." % best_model_name)
            for step in best_vclf3_estimator.steps:
                print(type(step))
                print("step:", step[0])
                params = step[1].get_params()
                for param_name in sorted(params.keys()):
                    print("\t%s: %r" % (param_name, params[param_name]))
            """

            if Y_type == 'binary':

                eu.plot_calibration_curves(
                    y_test, best_model_name + '_' + tuning_method,
                    models_data, 1, d_name
                    )

                plt.show()

            print()

            plt.close()

        else:

            # tuning and eventually calibrating
            # any other model != GaussianNB and LogReg

            m = re.search("Bagging_", best_model_name)
            if m:
                bagging_param_grid = \
                all_models_and_parameters['Bagging_' + best_model_name][1]

                if best_model_name == 'Bagging_SVMClf_2nd':
                    # 'Bagging_SVMClf_2nd' made of 10 estimators
                    del bagging_param_grid['n_estimators']
                
                param_grid = {
                    best_model_name + '__'
                    + k: v for k, v in bagging_param_grid.items()
                    }
            else:
                param_grid = all_models_and_parameters[best_model_name][1]

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
                random_state=random_state)

            temp_pipeline.set_params(**best_parameters)

            # plot a learning curve

            print()
            print()
            print("[task] === plotting a learning curve")

            y_lim = None
            if Y_type == "binary":
                y_lim = (0.5, 1.01)

            l_curve = 0
            try:
                au.plot_learning_curve(
                    temp_pipeline, X_train, y_train, ylim=y_lim, cv=kfold,
                    scoring=scoring, n_jobs=-2, serial=serial,
                    tuning=tuning_method, d_name=d_name
                    )

                plt.show()
            except JoblibValueError as jve:
                print("Not able to complete learning process...")
            except ValueError as ve:
                print(ve)
            except Exception as e:
                print(e)
            else:
                l_curve = 1

                del temp_pipeline
                temp_pipeline = training_pipeline

                print()
            finally:
                if not l_curve:
                    print("Sorry. Learning Curve plotting failed.")
                    print()

            ###

            print("Check '%s''s prediction confidence after (%s) CV and "
                  "calibrate probabilities."
                  % (best_model_name, tuning_method))
            print()

            # check predicted probabilities for prediction confidence
            uncalibrated_data = eu.probability_confidence_before_calibration(
                temp_pipeline, X_train, y_train, X_test, y_test, tuning_method,
                models_data, classes, serial
                )

            del temp_pipeline

            rscv_pred_score = uncalibrated_data[0]
            rscv_pipeline = uncalibrated_data[1]

            has_roc = 0
            try:
                uncalibrated_data[2]
            except IndexError:
                print("ROC_AUC score is not available.")
            except Exception as e:
                print(e)
            else:
                rscv_roc_auc = uncalibrated_data[2]
                has_roc = 1
            finally:
                print("We have target data to compare prediction confidence "
                      "of models.")
                if has_roc:
                    print("ROC_AUC included.")

            print()
            predicted = rscv_pipeline.predict(X_test)

            print()
            print("Before probability calibration...")

            if Y_type == 'binary':
                tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
                print()
                print("Test errors for '%s'" % best_model_name)
                print("\ttrue negatives: %d" % tn)
                print("\tfalse positives: %d" % fp)
                print("\tfalse negatives: %d" % fn)
                print("\ttrue positives: %d" % tp)
            else:
                print()
                print("Confusion matrix for '%s'.\n"
                      % best_model_name, confusion_matrix(y_test, predicted))
                print()
            print("Classification report for  '%s'\n"
                  % best_model_name, classification_report(y_test, predicted))
            print()

            # input("Enter key to continue... \n")
            print()

            # in case of LogRegression,
            # you should compare its probability curve against the ideal one

            if rscv_pred_score < lr_pred_score:
                print("'%s' is already well calibrated." % best_model_name)
                print("Let's resume metrics on test data.")

                w_rscv_acc = rscv_pipeline.score(X_test, y_test, sample_weight=w)*100

                print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f'
                      % (scoring.strip('neg_'), best_model_name, best_score))
                if has_roc:
                    print('Scoring [%s] of best uncalibrated ("%s") on test data: %1.3f'
                          % (scoring.strip('neg_'), best_model_name, rscv_roc_auc))
                print('Accuracy of best uncalibrated ("%s") on test data: %.2f%%'
                      % (best_model_name, w_rscv_acc))
                print()

                print("=== [task] Refit '%s' on all data." % best_model_name)
                print()
                print("X shape: ", X.shape)
                print("y shape: ", y.shape)
                print()

                # best_rscv_pipeline -> final_best_rscv_estimator
                final_best_rscv_pipeline = eu.rscv_tuner(
                    final_pipeline, X, y, n_splits, param_grid, n_iter,
                    scoring, refit=True, random_state=random_state
                    )

                print()
                print("Best estimator [%s]'s' params after hyp-tuning on all data."
                      % best_model_name)
                params = final_best_rscv_pipeline.get_params()
                for param_name in sorted(params.keys()):
                        print("\t%s: %r" % (param_name, params[param_name]))
                print()

                # input("Press key to continue... \n")

                w_best_rscv_acc = final_best_rscv_pipeline.score(
                    X, y, sample_weight=w_all)*100

                au.save_model(
                    final_best_rscv_pipeline, best_model_name
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
                print()

                temp_pipeline = training_pipeline

                # In case model needs calibration
                calib_data = eu.calibrate_probabilities(
                    temp_pipeline, X_train, y_train, X_test, y_test, 'sigmoid',
                    tuning_method, models_data, kfold, serial
                    )

                calib_rscv_pred_score = calib_data[0]
                calib_rscv_pipeline = calib_data[1]

                has_calib_roc = 0
                try:
                    calib_data[2]
                except IndexError:
                    print("ROC_AUC score is not available.")
                except Exception as e:
                    print(e)
                else:
                    calib_rscv_roc_auc = calib_data[2]
                    has_calib_roc = 1
                finally:
                    if has_calib_roc:
                        print("We have ROC_AUC for calibrated best model.")

                if calib_rscv_pred_score >= rscv_pred_score:
                    print("Sorry, we could not calibrate '%s' any better."
                          % best_model_name)
                    print("Rejecting calibrated '%s' and saving the uncalibrated one."
                          % best_model_name)

                    w_rscv_acc = rscv_pipeline.score(X_test, y_test, sample_weight=w)*100

                    print("Let's resume metrics on test data.")
                    print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f'
                          % (scoring.strip('neg_'), best_model_name, best_score))
                    if has_calib_roc:
                        print('Scoring [%s] of best uncalibrated ("%s") on test data: %1.3f%%'
                              % (scoring.strip('neg_'), best_model_name, calib_rscv_roc_auc))
                    print('Accuracy of best uncalibrated ("%s") on test data: %.2f%%'
                          % (best_model_name, w_rscv_acc))
                    print()

                    # final_best_rscv_estimator
                    final_best_rscv_pipeline = eu.rscv_tuner(
                        final_pipeline, X, y, n_splits, param_grid, n_iter,
                        scoring, refit=True, random_state=random_state)

                    w_best_rscv_acc = final_best_rscv_pipeline.score(
                        X, Y, sample_weight=w_all)*100

                    print('Accuracy of best  ("%s") on all data: %.2f%%'
                          % (best_model_name, w_best_rscv_acc))
                    print()

                    au.save_model(
                        final_best_rscv_pipeline, best_model_name
                        + '_final_nocalib_' + tuning_method + '_'
                        + serial + '.pkl', d_name=d_name)

                    print()

                    # Uncomment to see pipeline, steps and params
                    """
                    print("Finalized uncalibrated best model '%s'." % best_model_name)
                    for step in final_best_rscv_pipeline.steps:
                        print(type(step))
                        print("step:", step[0])
                        params = step[1].get_params()
                        for param_name in sorted(params.keys()):
                            print("\t%s: %r" % (param_name, params[param_name]))
                    print()
                    """

                else:
                    print("Achieved better calibration of model '%s'."
                          % best_model_name)
                    print("Let's resume scors on validation and test data.")
                    print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f'
                          % (scoring.strip('neg_'), best_model_name, best_score))

                    w_calib_rscv_acc = calib_rscv_pipeline.score(
                        X_test, y_test, sample_weight=w)*100

                    try:
                        calib_rscv_roc_auc
                    except NameError:
                        print("ROC_AUC score is not available.")
                    except Exception as e:
                        print(e)
                    else:
                        print('Scoring [%s] of best calibrated ("%s") on test data: %1.3f'
                              % (scoring, best_model_name, calib_rscv_roc_auc))
                    finally:
                        pass

                    print('Accuracy of best calibrated ("%s") on test data: %.2f%%'
                          % (best_model_name, w_calib_rscv_acc))
                    print()

                    print("=== [task]: Tune '%s'' params with '%s' on all "
                          "data and calibrate probabilities."
                          % (best_model_name, tuning_method))
                    print()

                    best_rscv_parameters = eu.rscv_tuner(
                        final_pipeline, X, y, n_splits, param_grid, n_iter,
                        scoring, refit=False, random_state=random_state
                        )

                    temp_pipeline.set_params(**best_rscv_parameters)
                    # calib_rscv_pipeline.named_steps[name].set_params(**best_parameters)

                    final_calib_rscv_clf = CalibratedClassifierCV(
                        temp_pipeline, method='sigmoid', cv=kfold)
                    final_calib_rscv_clf.fit(X, y)

                    fin_w_rscv_acc = final_calib_rscv_clf.score(
                        X, y, sample_weight=w_all)*100
                    print('Overall accuracy of finalized best CCCV ("%s_rscv"): %.2f%%'
                          % (best_model_name, fin_w_rscv_acc))

                    au.save_model(
                        final_calib_rscv_clf, best_model_name + '_final_calib_'
                        + tuning_method + '_' + serial + '.pkl', d_name=d_name)

            # Y_type = mc.type_of_target(Y)

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

        l_curve = 0
        try:
            au.plot_learning_curve(
                lr_pipeline, X_train, y_train, ylim=y_lim, cv=kfold,
                scoring=scoring, n_jobs=-2, serial=serial,
                tuning=tuning_method, d_name=d_name
                )

            plt.show()
        except JoblibValueError as jve:
            print("Not able to complete learning process...")
        except ValueError as ve:
            print(ve)
        except Exception as e:
            print(e)
        else:
            l_curve = 1

            del lr_pipeline
            lr_pipeline = uncalibrated_lr_data[1]

            print()
        finally:
            if not l_curve:
                print("Sorry. Learning Curve plotting failed.")
                print()

        print()
        print("'%s' is already well calibrated for definition!"
              % best_model_name)
        print()

        print('Mean cv score [%s] of best uncalibrated ("%s"): %1.3f'
              % (scoring.strip('neg_'), best_model_name, best_score))

        # best_lr_pipeline = lr_pipeline

        has_lr_roc = 0
        try:
            lr_roc_auc
        except NameError:
            print("ROC_AUC score is not available.")
        except Exception as e:
            print(e)
        else:
            best_lr_roc_auc = lr_roc_auc
            has_lr_roc = 1

            # print("ROC_AUC score on left-out data: %1.3f." % lr_roc_auc)
            # print("- The higher, the better.")
        finally:
            pass

        w_lr_acc = lr_pipeline.score(X_test, y_test, sample_weight=w)*100

        if has_lr_roc:
            print('Scoring [%s] of best uncalibrated ("%s") on test data: %1.3f'
                  % (scoring, best_model_name, best_lr_roc_auc))
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
        """
        print("Finalized '%s'." % best_model_name)
        for step in final_best_lr_pipeline.steps:
            print("step:", step[0])
            params = step[1].get_params()
            for param_name in sorted(params.keys()):
                print("\t%s: %r" % (param_name, params[param_name]))
        print()
        """

        if Y_type == 'binary':
            eu.plot_calibration_curves(
                y_test, best_model_name + '_rscv', models_data, 
                1, d_name)

            plt.show()

    plt.close('all')

    print()
    print()
