"""Building methods for evaluation and saving of classifiers."""

from autoclf.encoding import labelenc as lc

from time import time
import errno
import re
from random import randint
import numpy as np
from scipy.stats import normaltest
# from statsmodels.stats.stattools import jarque_bera
from sklearn.utils import class_weight as cw
from sklearn.utils import multiclass as mc

from . import param_grids_distros as pgd

from tempfile import mkdtemp
from shutil import rmtree

import matplotlib.pyplot as plt

from .. import auto_utils as au

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
try:
    from skopt import BayesSearchCV
except ImportError as ie:
    print(ie)
    print("Install scikit-optimize package if you want to use BayesCV")
from sklearn.base import is_classifier

from sklearn.externals.joblib.my_exceptions import JoblibValueError

from sklearn.pipeline import Pipeline
# from sklearn.pipeline import FeatureUnion

# for feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve

from sklearn.metrics import accuracy_score
# only for sklearn>=0.20:
# from sklearn.metrics import balanced_accuracy
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.wrappers.scikit_learn import KerasClassifier
sys.stderr = stderr

import warnings
warnings.filterwarnings("ignore")


# Methods

def calculate_sample_weight(dy):
    """Calculate sample weights in absence of class weights."""
    w = np.ones(dy.shape[0])
    w = cw.compute_sample_weight('balanced', dy)
    w = w/w.min()
    return w


def model_finalizer(
        estimator, data_X, data_Y, scoring, tuning, d_name=None, serial=0):
    """Train best estimator with all data and save it."""
    if (isinstance(scoring, list) or isinstance(scoring, dict)
            or isinstance(scoring, tuple)):
        raise TypeError(
            "'model_finalizer' method takes only single-metric score values.")
    if scoring not in (None, 'accuracy', 'roc_auc', 'neg_log_loss'):
        raise ValueError("""
            %s is not a valid scoring value for method 'model_finalizer'.
            Valid options are ['accuracy', 'roc_auc', 'neg_log_loss']"""
                         % scoring)

    name = ''
    for step in estimator.steps:
        m = re.search("Clf", step[0])
        if m:
            name = step[0]
    print("Finalized model refitted on all data")

    if hasattr(estimator, 'n_jobs'):
        if (hasattr(estimator, 'solver') and
                estimator.get_params()['solver'] == 'liblinear'):
            pass
        else:
            estimator.named_steps[name].set_params(n_jobs=-2)

    estimator.fit(data_X, data_Y)

    predicted = estimator.predict(data_X)
    if hasattr(estimator, "predict_proba"):
        predicted_probas = np.array(estimator.predict_proba(data_X))
        y_prob = []
        # consider label '0' --> 'dead'; label '1' --> 'survived'
        for probas in predicted_probas:
            y_prob.append(probas[1])
        print("predicted probabilities:\n", predicted_probas[0:3])
    else:
        # use decision function
        print("Model '%s' hasn't got 'predict_proba' as an attribute" % name)
        y_prob = estimator.decision_function(data_X)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    w = calculate_sample_weight(data_Y)

    print("Final scores on all data -- only for comparison purpose.")
    print("Accuracy %.2f%% computed with .score()"
          % (estimator.score(data_X, data_Y, sample_weight=w)*100))
    print("Accuracy %.2f%% computed with metrics.accuracy_score()"
          % (accuracy_score(data_Y, predicted, sample_weight=w)*100))
    if mc.type_of_target(data_Y) == 'binary':
        print("Scoring [%s] of model %s: %1.3f"
              % (scoring, name, roc_auc_score(data_Y, y_prob)))
    print()

    finalized_estimator = estimator

    print("Finalized best '%s'." % name)
    for step in finalized_estimator.steps:
        print(type(step))
        print("step:", step[0])
        params = step[1].get_params()
        for param_name in sorted(params.keys()):
            print("\t%s: %r" % (param_name, params[param_name]))

    if not serial:
        serial = "%04d" % randint(0, 1000)

    model_name = name

    if d_name is not None:
        name = d_name + "_" + name

    f_name = name + '_' + tuning + '_' + serial

    if model_name not in (
        'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
        'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'larger_deep_nn_Clf_2nd', 
        'deeper_nn_Clf_2nd', 'KerasClf_2nd'):
        au.save_model(finalized_estimator, f_name + '.pkl')
    else:
        keras_f_name = au.create_keras_model_filename(f_name)
        estimator.named_steps[model_name].model.save(keras_f_name + '.h5')

        if(len(estimator.steps)) > 1:
            estimator.named_steps[model_name].model = None
            f_name = name + '_' + tuning + '_for_keras_model_' + serial

            au.save_model(estimator, f_name + '.pkl')

    del estimator

    print()

    return finalized_estimator


def rscv_tuner(
        estimator, dx, dy, n_splits, param_grid, n_iter, scoring, refit=False, 
        cv_meth='rscv', random_state=0):
    """Tune best estimator before finalization."""
    if (isinstance(scoring, list) or isinstance(scoring, dict)
            or isinstance(scoring, tuple)):
        raise TypeError(
            "'rscv_tuner' method takes only single-metric score values.")
    if scoring not in (None, 'accuracy', 'roc_auc', 'neg_log_loss'):
        raise ValueError("""
            %s is not a valid scoring value for method 'rscv_tuner'.
            Valid options are ['accuracy', 'roc_auc', 'neg_log_loss']"""
                         % scoring)
    name = ''
    for step in estimator.steps:
        m = re.search("Clf", step[0])
        if m:
            name = step[0]

    kfold = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)

    xscv_prefix=None

    if cv_meth == 'rscv':
        clf = RandomizedSearchCV(
            estimator, param_distributions=param_grid, cv=kfold, iid=False, 
            n_iter=n_iter, n_jobs=-2, pre_dispatch="2*n_jobs", scoring=scoring, 
            refit=refit, random_state=random_state, error_score=np.nan)

        xscv_prefix='Randomized'

    elif cv_meth == 'bscv':
        # check 'params' is of type BayesSearchCV's search_spaces
        clf = BayesSearchCV(
            estimator, search_spaces=param_grid, iid=False, cv=kfold, n_iter=n_iter, 
            scoring=scoring, random_state=random_state, error_score=np.nan)

        xscv_prefix='Bayes'

    else:
        raise ValueError("Error. Valid values are ['rscv', 'bscv']")

    print("[task] === Hyperparameter tuning of %s with '%sSearchCV'"
          % (name, xscv_prefix))

    try:
        clf.fit(dx, dy)
    except TypeError as te:
        print(te)
        if hasattr(dx, 'values') is True:
            dx = dx.values
        if hasattr(dy, 'values') is True:
            dy = dy.values
        clf.fit(dx, dy)
    except Exception as e:
        raise e

    return clf.best_params_ if not refit else clf.best_estimator_


# works only for single-metric evaluation
def tune_and_evaluate(
        estimator, dx_train, dy_train, dx_test, dy_test, splits, param_grid,
        n_iter, scoring, models_data, refit, random_state, serial=0, 
        d_name=None, save=False, cv_meth='rscv'):
    """Tune and evaluate estimator with the options of refit and saving it."""
    if (isinstance(scoring, list) or isinstance(scoring, dict)
            or isinstance(scoring, tuple)):
        raise TypeError("""
            'tune_and_evaluate' method allows only to perform single-metric
            evaluation of given estimator.
            """)
    if scoring not in (None, 'accuracy', 'roc_auc', 'neg_log_loss'):
        raise ValueError("""
            %s is not a valid scoring value for method 'tune_and_evaluate'.
            Valid options are ['accuracy', 'roc_auc', 'neg_log_loss']"""
                         % scoring)

    name = ''

    if isinstance(estimator, Pipeline) and hasattr(estimator, "steps"):
        for step in estimator.steps:
            m = re.search("Clf", step[0])
            if m:
                if is_classifier(step[1]) or re.search("KerasClf", step[0]):
                    name = step[0]
                else:
                    raise TypeError(
                        "%r is not a valid classifier;\n" 
                        "a valid estimator is of type 'ClassifierMixin'" 
                        % step[1])
    elif is_classifier(estimator):
        name = 'classifier'
        param_grid = {
            k.split('__')[1]: v for k, v in param_grid.items()}
    else:
        raise TypeError(
            "%r is not a valid estimator for classification; valid estimators are\n"
            "of type ['Pipeline', 'ClassifierMixin']" % estimator)

    # kfold = KFold(n_splits=10, random_state=random_state)
    kfold = StratifiedKFold(
       n_splits=splits, shuffle=True, random_state=random_state)
    
    try:
        if cv_meth=='rscv':
            clf_name = 'RandomizedSearchCV'

            clf = RandomizedSearchCV(
                estimator, param_distributions=param_grid, cv=kfold, iid=False, 
                n_iter=n_iter, n_jobs=-2, pre_dispatch='2*n_jobs', scoring=scoring, 
                refit=refit, random_state=random_state)

        elif cv_meth=='bscv':
            clf_name = 'BayesSearchCV'

            clf = BayesSearchCV(
                estimator, search_spaces=param_grid, cv=kfold, iid=False, 
                n_iter=n_iter, n_jobs=-2, pre_dispatch='2*n_jobs', scoring=scoring, 
                refit=refit, random_state=random_state)
        else:
            raise ValueError("valid values are ['rscv', 'bscv']")

    except TypeError as te:
        print(te)
    except ValueError as ve:
        print(ve)
    except NameError as ne:
        print(ne)
    except Exception as e:
        raise e

    print()
    print("[task] === Hyperparameter Tuning %s with '%s'" % (name, clf_name))
    # automatize this based on search mode
    # if refit==True:
    tuning = 'rscv'

    try:
        t0 = time()
        # you do not need train-test splitting when doing cross validation
        clf.fit(dx_train, dy_train)
        t1 = time()
    except JoblibValueError as jve:
        t1 = time()
        print('%s -- JobLibValueError. Total execution time: %.2fs.'
              % (name, t1 - t0))
        print(jve)
    except ValueError as ve:
        t1 = time()
        print('%s -- ValueError. Total execution time: %.2fs.'
              % (name, t1 - t0))
        print(ve)
        print()
    except Exception as e:
        raise e
    else:
        print("Execution time after '%s': %.2fs." % (clf_name, t1 - t0))

    if refit:
        dy_type = mc.type_of_target(dy_train)

        predicted = clf.predict(dx_train)

        print("=== '%s''s performance & accuracy of predictions" % name)

        if scoring is None:
            ret_scoring = 'accuracy'
        else:
            ret_scoring = scoring

        rscv_score = clf.score(dx_train, dy_train)
        mean_rscv_score = clf.best_score_

        if scoring == 'neg_log_loss':
            rscv_score = -1.*rscv_score
            mean_rscv_score = -1.*mean_rscv_score

        print("Scoring [%s] %1.3f computed with rscv.score()"
              % (ret_scoring.strip('neg_'), rscv_score))
        print("Mean cv score [%s] of the best_estimator: %1.3f"
              % (ret_scoring.strip('neg_'), mean_rscv_score))

        if dx_test is not None:
            y_pred = clf.predict(dx_test)

            print("=== Predictions with %s after %s" % (name, clf_name))
            print("target:\n", dy_train[0:5])
            print("predictions:\n", predicted[0:5])

            if hasattr(clf, "predict_proba"):
                predicted_probas = np.array(clf.predict_proba(dx_test))
                if dy_type == "binary":
                    y_prob = []
                    # consider label '0' --> 'negative'; label '1' --> 'positive'
                    for probas in predicted_probas:
                        y_prob.append(probas[1])
                else:
                    y_prob = predicted_probas
                print("predicted probabilities before calibration:\n",
                        predicted_probas[0:3])
            else:
                # use decision function
                print("Model '%s' hasn't got 'predict_proba' as an attribute"
                        % clf)
                y_prob = clf.decision_function(dx_test)
                y_prob = (y_prob - y_prob.min())/(y_prob.max() - y_prob.min())

            w = calculate_sample_weight(dy_test)

            print("Accuracy of predictions on new data %.2f%%" % (
                accuracy_score(dy_test, y_pred, sample_weight=w)*100))
            if dy_test.max():
                # consider label '0' --> negative class;
                # label '1' --> positive class
                log_loss_score = log_loss(dy_test, y_prob)
                brier_score = brier_score_loss(dy_test, y_prob, pos_label=1)
                print("=== Predicted probabilities confidence for '%s'" % name)
                print("log loss for '%s': %1.3f" % (name, log_loss_score))
                print("brier score: %1.3f" % brier_score)
                print()

            print("Confusion matrix for %s after %s\n" % (name, clf_name),
                  confusion_matrix(dy_test, y_pred))
            print()
            print("Classification report for %s after %s\n" % (name, clf_name),
                  classification_report(dy_test, y_pred))
            print()

            got_preds = 0
            try:
                models_data.append((f_name, y_prob, y_pred))
                got_preds = 1
            except Exception as e:
                print(e)
            finally:
                if got_preds == 1:
                    pass
                else:
                    print("Failed to add predictions to models_data")

        best_clf = clf.best_estimator_

        if name in ("RandomForestClf_2nd", "ExtraTreesClf_2nd"):
                print("Out-of-bag score estimate: %1.3f"
                      % best_clf.named_steps[name].oob_score_)

        # report(clf.cv_results_)
        # print()

        if save:
            saved = 0

            if not serial:
                serial = "%04d" % randint(0, 1000)    # %04d

            f_name = name + '_refitted_' + tuning + '_' + serial

            if d_name is not None:
                f_name = d_name + "_" + f_name

            try:
                if name not in (
                    'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
                    'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'larger_deep_nn_Clf_2nd', 
                    'deeper_nn_Clf_2nd', 'KerasClf_2nd'):
                    au.save_model(best_clf, f_name + '.pkl')
                else:
                    saved_estimator = best_clf
                    keras_f_name = au.create_keras_model_filename(f_name)
                    saved_estimator.named_steps[name].model.save(keras_f_name + '.h5')

                    if(len(saved_estimator.steps)) > 1:
                        saved_estimator.named_steps[name].model = None
                        f_name = name + '_' + tuning + '_for_keras_model_' + serial

                        au.save_model(saved_estimator, f_name + '.pkl')
                saved = 1
            except Exception as e:
                print(e)
            finally:
                if saved == 1:
                    pass
                else:
                    print("Failed to save model to file")

        else:
            pass

    else:
        # refit == False
        print("=== '%s''s performance without predictions" % name)

        try:
            mean_rscv_score = clf.best_score_
        except AttributeError as ae:
            print(ae)
        except Exception as e:
            raise e
        else:
            if scoring == 'neg_log_loss':
                mean_rscv_score = -1.*mean_rscv_score

            print("Mean cv score [%s] of the best_estimator: %1.3f"
                  % (scoring.strip('neg_'), mean_rscv_score))

    print()
    print("Parameters of best estimator.")
    for param_name in sorted(clf.best_params_.keys()):
        print("\t%s: %r" % (param_name, clf.best_params_[param_name]))
    print()

    return clf.best_params_ if not refit else best_clf


def probability_confidence_before_calibration(
        unc_estimator, dx_train, dy_train, dx_test, dy_test, tuning,
        models_data, labels, serial=0, save=False):
    """Train estimator to check prediction confidence before calibration."""
    print()
    print("[task] === Probability confidence before calibration")

    name = ''
    for step in unc_estimator.steps:
        m = re.search("Clf", step[0])
        if m:
            name = step[0]
            model = step[1]

    # speed things up, use more CPU jobs
    if hasattr(model, 'n_jobs'):
        if (hasattr(model, 'solver') and
                model.get_params()['solver'] == 'liblinear'):
            pass
        else:
            unc_estimator.named_steps[name].set_params(n_jobs=-2)

    unc_estimator.fit(dx_train, dy_train)
    # best_clf = unc_estimator.named_steps[name]

    dy_type = mc.type_of_target(dy_test)

    w = calculate_sample_weight(dy_test)
    wtr = calculate_sample_weight(dy_train)

    # if hasattr(unc_estimator, "predict"):
    predicted = unc_estimator.predict(dx_test)

    print()
    print("=== Predictions with %s after %s" % (name, tuning.upper()))
    print("target:\n", dy_test[0:5])
    print("predictions:\n", predicted[0:5])
    print("=== '%s''s quality of predictions" % name)

    print("Accuracy %.2f%% computed with .score() on train data"
          % (unc_estimator.score(dx_train, dy_train, sample_weight=wtr)*100))
    print("Accuracy %.2f%% computed with metrics.accuracy_score() on new data"
          % (accuracy_score(dy_test, predicted, sample_weight=w)*100))

    if hasattr(unc_estimator, "predict_proba"):
        predicted_probas = np.array(unc_estimator.predict_proba(dx_test))
        if dy_type == "binary":
            y_prob = []
            # consider label '0' --> 'dead'; label '1' --> 'survived'
            for probas in predicted_probas:
                y_prob.append(probas[1])
        else:
            y_prob = predicted_probas
        # print("predicted probabilities:\n", predicted_probas[0:3])
        print("predicted probabilities:\n", y_prob[0:3])
    else:
        # use decision function
        print("Model '%s' hasn't got 'predict_proba' as an attribute" % name)
        y_prob = unc_estimator.decision_function(dx_test)
        y_prob = (y_prob - y_prob.min())/(y_prob.max() - y_prob.min())
    if dy_type == "binary":
        unc_roc_auc = roc_auc_score(dy_test, y_prob)
        print("ROC_AUC score computed on new_data: %1.3f" % unc_roc_auc)
    print("=== Predicted probabilities confidence for '%s'" % name)
    if labels is not None:
        print("Labels:", labels)
    else:
        print("Labels is 'None'")
    prediction_confidence = log_loss(dy_test, y_prob, labels=labels)
    print("log loss for '%s': %1.3f" % (name, prediction_confidence))
    if dy_type == "binary":
        print("brier score: %1.3f"
              % brier_score_loss(dy_test, y_prob, pos_label=1))

    if not serial:
        serial = "%04d" % randint(0, 1000)

    f_name = name + '_nocalib_' + tuning + '_' + serial

    if save:
        saved = 0
        try:
            if name not in (
                'baseline_nn_default_Clf_2nd', 'baseline_nn_smaller_Clf_2nd',
                'larger_nn_Clf_2nd', 'deep_nn_Clf_2nd', 'larger_deep_nn_Clf_2nd', 
                'deeper_nn_Clf_2nd', 'KerasClf_2nd'):
                au.save_model(unc_estimator, f_name + '.pkl')
            else:
                saved_estimator = unc_estimator
                keras_f_name = au.create_keras_model_filename(f_name)
                saved_estimator.named_steps[name].model.save(keras_f_name + '.h5')

                if(len(saved_estimator.steps)) > 1:
                    saved_estimator.named_steps[name].model = None
                    f_name = name + '_' + tuning + '_for_keras_model_' + serial

                    au.save_model(saved_estimator, f_name + '.pkl')

            del saved_estimator

            saved = 1
        except Exception as e:
            print(e)
        finally:
            if saved == 1:
                pass
            else:
                print("Failed to save model to file")

    got_preds = 0
    try:
        models_data.append((f_name, y_prob, predicted))
        got_preds = 1
    except Exception as e:
        print(e)
    finally:
        if got_preds == 1:
            pass
        else:
            print("Failed to add predictions to models_data")

    print()

    unc_est_data = ()

    # back to n_jobs=1 in order to avoid nesting parallel jobs into RSCV
    if hasattr(model, 'n_jobs'):
        if (hasattr(model, 'solver') and
                model.get_params()['solver'] == 'liblinear'):
            pass
        else:
            unc_estimator.named_steps[name].set_params(n_jobs=1)

    if dy_type == "binary":
        unc_est_data = prediction_confidence, unc_estimator, unc_roc_auc
    else:
        unc_est_data = prediction_confidence, unc_estimator

    return unc_est_data


def calibrate_probabilities(
        estimator, dx_train, dy_train, dx_test, dy_test, method, tuning,
        models_data, cv, labels, serial=0):
    """Calibrate probabilities of 'estimator'."""
    if method not in ('isotonic', 'sigmoid'):
        raise ValueError("""%s is not a valid method value for function
        'calibrate_probabilities_prod'. Valid options are ['isotonic',
        'sigmoid']""" % method)

    name = ''
    try:
        for step in estimator.steps:
            m = re.search("Clf", step[0])
    except Exception as e:
        print(e)
    else:
        if m:
            name = step[0]

    print("[task] === Calibrating predicted probabilities.")
    print()

    if len(dy_train) < 100:
        if method == 'isotonic':
            print("""It is not advised to use isotonic calibration with too few
            calibration samples since it tends to overfit.""")
            print("Switching to sigmoids (Platt's calibration).")
            method = 'sigmoid'

    elif (len(dy_train) >= 100000 or
          dx_train.shape[1]*dx_train.shape[0] >= 100000):
        print("Too many samples or too much data, using only isotonic calibration.")
        if method == 'sigmoid':
            method = 'isotonic'

    dy_type = mc.type_of_target(dy_test)

    calib_clf = CalibratedClassifierCV(estimator, method=method, cv=cv)

    # if cv == 'prefit': train data (dx_train, dy_train) == (X_calib, y_calib))
    # else: train data (dx_train, dy_train) == (X_train, Y_train)

    t0 = time()
    calib_clf.fit(dx_train, dy_train)
    t1 = time()
    wtr = calculate_sample_weight(dy_train)
    w_score = calib_clf.score(dx_train, dy_train, sample_weight=wtr)*100

    print("Execution time after '%s' 'CCCV': %.2fs." % (method, t1 - t0))
    predicted = calib_clf.predict(dx_test)

    # print("predicted:\n", predicted_probas[0:3])

    print("=== Predictions after calibration for '%s'" % name)
    print("predictions:\n", predicted[0:3])
    # print("predicted probabilities:\n", predicted_probas[0:3])
    print("=== Calibration performance for '%s'" % name)
    prediction_score = 10000

    if dy_type == 'binary':

        # consider label '0' --> 'dead'; label '1' --> 'survived'
        predicted_probas = np.array(calib_clf.predict_proba(dx_test))
        y_prob = []
        # consider label '0' --> 'dead'; label '1' --> 'survived'
        for probas in predicted_probas:
            y_prob.append(probas[1])

        clabel = 1
        print("predicted probabilities for class %d:\n" % clabel, y_prob[0:5])
        bsl_score = brier_score_loss(dy_test, y_prob, pos_label=clabel)
        print("Brier score: %1.3f" % bsl_score)
        print("=== '%s''s classification performance" % name)
        # acc_from_predictions = accuracy_score(dy_test, predicted)*100
        roc_auc_fom_preds = roc_auc_score(dy_test, y_prob)*100
        print("ROC AUC %1.3f computed w metrics.roc_auc_score() on test data and predictions"
              % roc_auc_fom_preds)

        msg = "*** Best Brier scor loss: "
        prediction_score = bsl_score

    else:
        y_prob = np.array(calib_clf.predict_proba(dx_test))
        print("Predicted probabilities:\n", y_prob[0:5])

        ll_score = log_loss(dy_test, y_prob, labels=labels)
        print("log loss for '%s': %1.3f" % (name, ll_score))

        msg = "*** Best log loss score: "
        prediction_score = ll_score

    print("=== '%s''s accuracy of predictions" % name)
    w = calculate_sample_weight(dy_test)
    weighted_acc_from_predictions = accuracy_score(
        dy_test, predicted, sample_weight=w)*100

    print("Accuracy %.2f%% computed with .score()" % (w_score))
    print("Accuracy %.2f%% computed w metrics.accuracy_score() on test data "
          "and predictions" % weighted_acc_from_predictions)

    print(msg + "%1.3f, method: %s" % (prediction_score, method))

    print()

    got_preds = 0
    try:
        models_data.append(
            (name + '_nofinal_calib_' + tuning + '_' + method + '_'
             + serial, y_prob, predicted))
        got_preds = 1
    except Exception as e:
        print(e)
    finally:
        if got_preds == 1:
            pass
        else:
            print("Failed to add predictions to models_data")

    print()

    calib_est_data = ()

    if dy_type == 'binary':
        calib_est_data = prediction_score, calib_clf, roc_auc_fom_preds
    else:
        calib_est_data = prediction_score, calib_clf

    return calib_est_data


def plot_calibration_curves(dy_test, clf_name, models_data, fig_index, d_name=None):
    """Plot calibration curve for various models - w/o and with calibration."""
    print()
    print("=== Plotting calibration curves.")
    print("length of models_data:", len(models_data))

    # curdir = os.path.dirname(__file__)
    curdir = os.getcwd()

    # directory = curdir + "\\results"
    directory = os.path.join(curdir, "results")

    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    w = calculate_sample_weight(dy_test)

    # your models should already be fitted at this point
    # and associated predicted probabilities ready to be used
    # to assess performance and confidence of predictions

    if d_name is not None:
        clf_name = d_name + "_" + clf_name

    name_w_serial = ''

    for it, data in enumerate(models_data):
        print("it + 1 : ", (it + 1))

        name, prob_pos, y_pred = data

        w_clf_score = brier_score_loss(dy_test, prob_pos, pos_label=dy_test.max())
        print("%s weighted scores:" % name)
        print("\tBrier: %1.3f" % (w_clf_score))
        print("\tLogLoss: %1.3f" % log_loss(dy_test, prob_pos))
        print("\tAccuracy: %1.3f" % accuracy_score(dy_test, y_pred, sample_weight=w))
        print("\tROC AUC: %1.3f" % roc_auc_score(dy_test, prob_pos))
        print("\tPrecision: %1.3f" % precision_score(dy_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(dy_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(dy_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(dy_test, prob_pos, n_bins=10)

        ax1.plot(
            mean_predicted_value, fraction_of_positives, "s-",
            label="%s (%1.3f)" % (name, w_clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step",
                 lw=2)

        if (it + 1) == len(models_data):
            assert (it + 1) == len(models_data), "jebiga, you failed!"
            name_w_serial = name
            print(name_w_serial)

    try:
        re.search(r'(?<=_)\d{4}', name_w_serial).group(0)
    except AttributeError as ae:
        print(ae)
        serial = "%04d" % randint(0, 1000)
    except Exception as e:
        print(e)
    else:
        serial = re.search(r'(?<=_)\d{4}', name_w_serial).group(0)
        # clf_name = sys.argv[0][:5] + "_" + clf_name + "_" + serial
        clf_name = clf_name + "_" + serial

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots of "%s" (reliability curve)' % clf_name)

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

    try:
        fig_name = os.path.join(curdir, directory, "calibration_curves_" + clf_name)
    except FileNotFoundError as fe:
        print(fe)
    except Exception as e:
        print(e)
    else:
        plt.savefig(fig_name + ".png", format='png')


def scoring_and_tt_split(df, target, test_size=0.3, random_state=0, Xy=False):
    """
    # Select scoring based on type of target and train-test split data.

    ----------------------------------------------------------------------------
    df: pandas dataframe
    target: pandas series containing target label
    random_state: seed [np.random]
    """
    print()
    print("[task] === Select scoring based on type of target [supervised] and "
          "train-test split data")

    print()
    print(df[[target]].head())
    print()

    # Split dataframe in features and target

    X = df.drop([target], axis=1)
    y = df[target]

    print("X.columns:", X.columns)
    print("y elements:", y.unique())
    print()

    X = columns_as_type_float(X)

    # for col in X.columns:
    #     if X[col].dtype in ('uint8', 'int8', 'int32', 'int64'):
    #         X[col] = X[col].astype(float)

    print()
    # print("Before feature union.")
    print("=== Before feature transfomation.")
    print()
    print("X shape: ", X.shape)
    print("Y shape: ", y.shape)
    print()

    # Here you could ask whether to optimize for a single metric or
    # go for multimetric-evaluation

    labels = None

    Y_type = mc.type_of_target(y)

    if (Y_type not in ("binary", "multilabel-indicator") and
            Y_type == 'multiclass'):
        # nested_cv_scoring='accuracy'
        scoring = 'neg_log_loss'

        labels = list(np.unique(y))

    elif Y_type == "binary":
        scoring = 'roc_auc'

        print()

    print("Metric:", scoring.strip('neg_'))
    print("Calibration of untrained models -- CCCV 2nd")
    print()

    # Non-nested CV

    # Train-test split

    print()

    print("== First split: train-test")

    X_train, X_test, y_train, y_test = (None, None, None, None)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, Y, stratify=Y, test_size=test_size, random_state=random_state)

    train_idx, test_idx = (None, None)

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state)

    for train_index, test_index in sss.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        train_idx, test_idx = train_index, test_index

    # original dataframe may contain NaNs and Infs propagating
    # into train, test dfs

    if X_train.isnull().values.any():

        X_train = X_train.fillna(method='bfill')
        X_train = X_train.fillna(method='pad')

    if X_test.isnull().values.any():

        X_test = X_test.fillna(method='bfill')
        X_test = X_test.fillna(method='pad')

    sltt = {}

    sltt['target_type'] = Y_type
    sltt['labels'] = labels
    sltt['scoring'] = scoring
    sltt['tt_index'] = train_idx, test_idx
    sltt['arrays'] = X_train, X_test, y_train, y_test
    if Xy:
        sltt['Xy'] = X, y

    return sltt


# Processing df columns

def float_columns(X):
    for col in X.columns:
        if X[col].dtype in ('uint8','int8', 'int32', 'int64'):
            X[col] = X[col].astype(np.float32)

    return X


def columns_as_type_float(X):
    try:
        X = X.astype(np.float32)
    except MemoryError as me:
        print("MemoryError !")
        print(me)
        X = float_columns(X)
    except TypeError as te:
        print("TypeError !")
        print(te)
        X = float_columns(X)
    except ValueError as ve:
        print("ValueError !")
        print(ve)
        X = float_columns(X)
    except Exception as e:
        raise e

    return X


def reorder_ohencoded_X_test_columns(X_train, X_test):
    col_to_add = np.setdiff1d(
        X_train.columns, X_test.columns)

    # add these columns to test, setting them equal to zero
    for c in col_to_add:
        X_test[c] = 0

    # select and reorder the test columns using the train columns
    X_test_reordered = X_test[X_train.columns]

    assert (len(X_test_reordered.columns) == len(X_train.columns)), \
        "Number of columns do not match!"

    return X_test_reordered


def create_feature_selector(X, encoding, seed=0):
    if encoding not in ("le", "ohe"):
        raise ValueError(
            "%s is not a valid encoding value."
            "Valid values are ['le', 'ohe']")

    if encoding == "le":
        if X.shape[1] > 10:
                featselector = (
                    'feature_selection', SelectFromModel(
                        ExtraTreesClassifier(
                            n_estimators=10, random_state=seed))
                        )
        else:
            # threshold=0. bypasses selection
            featselector = (
                'feature_selection', SelectFromModel(
                    ExtraTreesClassifier(
                        n_estimators=10, random_state=seed), 
                        threshold=0.)
                        )
    else:
        lscv = pgd.full_search_models_and_parameters['LinearSVMClf_2nd'][0]
        lscv = lscv.set_params(C=0.01, penalty="l1", dual=False)
        # threshold=[1e-2, 1e-1]
        featselector = ('feature_selection', 
                        SelectFromModel(lscv, threshold="mean"))

    return featselector


def plot_feature_importances(estimator, X):
    sfm_est = estimator.named_steps['feature_selection'].estimator_

    feature_names = X.columns

    # importances = []

    try:
        if hasattr(sfm_est, 'feature_importances_'):
            importances = sfm_est.feature_importances_
            importances = 100*(importances/np.max(importances))
            fs = 'Tree-based feature selection'
        elif hasattr(sfm_est, 'coef_'):
            abs_importances = np.abs(sfm_est.coef_[0])
            importances = 100*(abs_importances/np.max(abs_importances))
            """
            if sfm_est.coef_.ndim == 1:
                importances = np.abs(sfm_est.coef_)
            else:
                importances = np.linalg.norm(
                    sfm_est.coef_, axis=0, ord=norm_order)
            """
            fs = 'L1-based feature selection'
    except Exception as e:
        print(e)
    else:
        indices = np.argsort(importances)[::-1]

        print("indices:", indices)
        print("indices shape:", indices.shape)
        print("X_train_encoded.shape[1]:", X.shape[1])
        print("feature_names shape:", feature_names.shape)
        print("importances shape", importances.shape)
        print()

        # Print the feature ranking - top 64
        title  = "Feature importances from " + fs

        x_length = X.shape[1]
        print("Feature ranking:")
        if x_length > 64:
            x_length = 64
            title  = "Top 64 feature importances from " + fs
        print(title)
        print()
        for f in np.arange(x_length):
            print("\t%d. feature '%s' (%.5f)"
                  % (f + 1,
                     feature_names[indices[f]],
                     importances[indices[f]]))
        print()

        # Plot the feature importances of the sfm.estimator - top 64
        pos = np.arange(x_length) + .5
        indices = np.argsort(importances)[:x_length]

        plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.barh(pos, importances[indices], color="b", align="center")
        plt.yticks(pos, feature_names[indices])
        plt.xlabel('Relative Importance')
        plt.show()

    print()
    # input("Press any key to continue...")
    plt.close()


# after that, you can inject your feature engineering on train-test data

def auto_X_encoding(
        sltt, random_state, f_sel=True, encode_target_fct=None):
    """
    # automatically label-encode or OH-encode train-test dataself.

    # plot feature importances of label-encoded data.

    ------------------------------------------------------

    sltt: dictionary of scoring, labels and train-test data
    random_state: seed [np.random]
    encode_target_fct: function to encode target
    """
    print()
    print("[task] === Automatically label-encode or OH-encode train-test data")

    X_train, X_test, y_train, y_test = sltt['arrays']

    # custom engineering of train, test dfs may creaet NaNs for missing values

    if X_train.isnull().values.any():

        X_train = X_train.fillna(method='bfill')
        X_train = X_train.fillna(method='pad')

    if X_test.isnull().values.any():

        X_test = X_test.fillna(method='bfill')
        X_test = X_test.fillna(method='pad')

    print()
    print("Before encoding:")
    print()
    # print("X_test -- first row:", X_test.values[0])
    # print("X_train -- first row:", X_train.values[0])
    # print()

    encoding = au.select_encoding_method()

    print()
    # input("Enter key to continue... \n")

    scaler_tuple = au.select_X_transformer()

    steps = []
    # here you should also insert imputing and label encoding
    # steps.append(('label_encoder', LabelEncoder()))
    steps.append(scaler_tuple)

    featselector = None

    if encoding == 'le':

        print("=== [task] Label-Encoding X_train and X_test dataframe values")
        print()

        print("Label-Encoding X_train")
        X_train_encoded = lc.dummy_encode(X_train).astype(np.float32)
        print()

        print("Label-Encoding X_test")
        X_test_encoded = lc.dummy_encode(X_test).astype(np.float32)

        featselector = create_feature_selector(
            X_train_encoded, encoding, random_state)

        print()
        print("X_test shape: ", X_test_encoded.shape)

    else:

        print("=== [task] One-Hot-Encoding X_train and X_test dataframe values")
        print()
        print("X_train info:\n", X_train.info())
        print()
        print("X_train's head: ", X_train.head(1))
        print()
        # (le) + ohe
        print("OH-Encoding X_train")

        try:
            X_train_encoded = lc.get_dummies_or_label_encode(X_train)
            print()
        except MemoryError as me:
            print("MemoryError !")
            print(me)
            print()
            print("Label-Encoding X_train")
            X_train_encoded = lc.dummy_encode(X_train).astype(np.float32)
            print()

            print("Label-Encoding X_test")
            X_test_encoded = lc.dummy_encode(X_test).astype(np.float32)

            encoding = 'le'
        except Exception as e:
            print("One-Hot-Encoding failed.")
            raise e
        else:
            # X_train_encoded =X_train_encoded.astype(np.float32)
            X_train_encoded = columns_as_type_float(X_train_encoded)
            print()
            
        print()

        if encoding == "le":
            featselector = create_feature_selector(
                X_train_encoded, encoding, random_state)
        else:
            # encoding == "ohe":
            print("OH-Encoding X_test")
            try:
                X_test_encoded = lc.get_dummies_or_label_encode(X_test)
                print()
            except Exception as e:
                raise e
            else:
                # X_test_encoded = X_test_encoded.astype(np.float32)
                X_test_encoded = columns_as_type_float(X_test_encoded)
                print()

                # perform feature selection by means of dimensionality reduction
                # to reduce data cardinality of oh-encoded data

                featselector = create_feature_selector(
                    X_train_encoded, encoding, random_state)

                X_test_encoded = reorder_ohencoded_X_test_columns(
                    X_train_encoded, X_test_encoded)

                print()
                print("After column trick, X_test shape: ", X_test_encoded.shape)

    print("X_test.columns:", X_test_encoded.columns)
    print("X_test -- first row:", X_test_encoded.values[0])
    print("y_test shape: ", y_test.shape)

    print("X_train shape: ", X_train_encoded.shape)
    print("X_train.columns:", X_train_encoded.columns)
    print("X_train -- first row:", X_train_encoded.values[0])
    print("y_train shape: ", y_train.shape)
    print()

    steps.append(featselector)
    pipeline = Pipeline(steps)

    if not f_sel:
        pipeline.fit(X_train_encoded, y_train)
    else:
        transformer = pipeline.fit(X_train_encoded, y_train)

    fs = ''

    if encoding == 'le':
        plot_feature_importances(pipeline, X_train_encoded)
        fs = 'ETR-based feature selection'
        
    else:
        # OHE
        fs = 'L1-based feature selection'

    auto_feat_eng_data = {}

    if not f_sel:
        auto_feat_eng_data['data_arrays'] = (
            X_train_encoded, y_train, X_test_encoded, y_test)
    else:
        X_train_transformed = transformer.transform(X_train_encoded)
        X_test_transformed = transformer.transform(X_test_encoded)

        print()
        print("=== After feature transfomation ['%s' encoding + %s]."
              % (encoding, fs))
        print()
        print("X_train_transformed shape: ", X_train_transformed.shape)
        print("X_test_transformed shape: ", X_test_transformed.shape)

        # print("X_train_transformed sample: ", X_train_transformed[:3])
        # print("X_test_transformed: ", X_test_transformed[:3])

        auto_feat_eng_data['fsel_transformer'] = transformer
        auto_feat_eng_data['data_arrays'] = (
            X_train_transformed, y_train, X_test_transformed, y_test)
        auto_feat_eng_data['tt_index'] = sltt['tt_index']

    X = X_train_encoded.append(X_test_encoded)
    y = y_train.append(y_test)
    auto_feat_eng_data['Xy'] = X, y

    auto_feat_eng_data['encoding'] = encoding
    auto_feat_eng_data['scaler'] = scaler_tuple
    auto_feat_eng_data['feat_selector'] = featselector
    auto_feat_eng_data['steps'] = steps

    return auto_feat_eng_data


def split_and_X_encode(dataframe, target, test_frac, random_state=0, Xy=False,):
    """Train test split data and label encode features."""
    sltt = scoring_and_tt_split(dataframe, target, test_frac, random_state, Xy)

    X_train, X_test, y_train, y_test = sltt['arrays']

    scoring = sltt['scoring']
    Y_type = sltt['target_type']
    classes = sltt['labels']

    print("Classes:", classes)

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

    # auto_feat_eng_data
    return auto_X_encoding(sltt, random_state), scoring, Y_type, classes


# Model evaluation

def calculate_stats_for_nhst(
        name, cv_results, scoring, robust=False):
    """calculate stats for statistical hypothesis testing"""
    invert_ineq = False

    if scoring == "neg_log_loss":
        invert_ineq = True

    if not robust:
        score = np.mean(cv_results)
        print("Using the mean of scores.")

        if not invert_ineq:
            func = 'x_mean > y_mean'
        else:
            func = 'x_mean < y_mean'

    if len(cv_results) >= 10:

        if robust:

            print("Using median and median absolute deviation of scores.")

            # median of cv scores
            score = np.median(cv_results)
            # median abs deviation of cv scores
            score_dev = au.mad(cv_results)

            if not invert_ineq:
                func = 'x_median > y_median'
            else:
                func = 'x_median < y_median'

        else:
            print("Using the standard deviation.")

            score_dev = np.std(cv_results)
            
    else:
        print("Too few samples, using half-dispersion of scores.")

        # half-dispersion
        score_dev = np.abs(np.ptp(cv_results))/2

    stats = score, score_dev, func

    return stats


# def test_for_normal_distribution(name, cv_results, alpha):
#     """Assess normality assumption to use parametric tests for nhst"""
        
#     # here, value == 'kurtosis'
#     _, p, _, kurtosis = jarque_bera(cv_results)

#     print("kurtosis: %1.3f; p_norm: %1.3f" % (kurtosis, p))

#     # null hypothesis: x comes from a normal distribution
#     normality = "likely"

#     if p < alpha:
#         # The null hypothesis can be rejected
#         print("It is unlikely that distro of cv_results for '%s' is normal.")

#         normality = "unlikely"

#     else:
#         # p >= alpha
#         # Fail to reject the null hypothesis
#         print("It is likely that distro of cv_results for '%s' is normal."
#               % name)

#     return normality


def compare_models_performance(
        name, model, exec_time, best_model_name, best_score_dev, stats, 
        cv_results, best_cv_results, average_scores_across_outer_folds, 
        scores_of_best_model, func, cv_style="xscv", scoring="roc_auc", 
        params=None, random_state=42):
    """statistical testing of classifiers"""

    score, score_dev = stats[0], stats[1]
    best_score, best_score_dev = stats[2], stats[3]

    print("=== Null hypothesis statistical testing")
    print("Null hypothesis: %s is worst / no better than %s"
          % (name, best_model_name))
    print("Alternative hypothesis: %s is better than %s"
          % (name, best_model_name))

    # cv_diff = cv_results - best_cv_results
    if scoring == 'brier_score_loss':
        # brier_score in [0, 1], the lower, the better
        cv_results, best_cv_results = 1-cv_results, 1-best_cv_results

    if cv_style == "xscv":
        try:
            params
        except AttributeError as ae:
            print(ae)
        except TypeError as te:
            print(te)
        except Exception as e:
            raise e
    elif cv_style == "classic":
        params = model.get_params()
    elif cv_style != "xscv":
        raise ValueError("%s in not a valid value for 'cv_style'"
                         "valid values are ['classic', 'xscv']" % cv_style)

    pt_method = 'exact'

    if len(cv_results) > 10:    
        pt_method = 'approximate'

    # calculate probability of difference due to chance
    p_value = au.permutation_test(
        cv_results, best_cv_results, func=func, method=pt_method,
        seed=random_state)

    print("P_value': %1.5f" % p_value)

    if p_value < 0.05:
        print("Reject the null hypothesis")
        print("It is unlikely that %s's performance is worst / no better."
            % name)

        msg = "Assuming that '%s' is a better classifier than '%s'."

        print(msg % (name, best_model_name))

        best_score = score
        best_score_dev = score_dev
        best_cv_results = cv_results
        best_exec_time = exec_time
        best_model_name = name
        best_model_estim = model
        if name in (
                'baseline_nn_default_Clf_2nd',
                'baseline_nn_smaller_Clf_2nd', 'larger_nn_Clf_2nd',
                'deep_nn_Clf_2nd', 'larger_deep_nn_Clf_2nd', 
                'deeper_nn_Clf_2nd', 'KerasClf_2nd'):
            best_nn_build_fn = model.get_params()['build_fn']
        else:
            best_nn_build_fn = None
        best_model = (best_model_name, best_model_estim, best_nn_build_fn)

        print("We have a new champion!")
        scores_of_best_model = \
            (best_score, best_score_dev, best_cv_results,
             best_exec_time, best_model)

        print("*** Now, best score [%s]: %1.3f, best model: %s"
              % (scoring.strip('neg_'), best_score, best_model_name))
        print()

        average_scores_across_outer_folds[name] = \
            (score, score_dev, exec_time, model, params)

    else:
        print("Fail to reject the null hypothesis")
        print("It is likely that %s' performance is worst or no better." % name)

    return average_scores_across_outer_folds, scores_of_best_model


def best_model_initial_attributes(scoring, n_splits):
    """Define initial best model's initial attributes"""
    if scoring in ('roc_auc', 'average_precision_score'):
        best_score = 0.5    # 0.0
        best_score_dev = 0.5
        best_cv_results = np.zeros(n_splits)
        best_model_name = 'Random'
    elif scoring == 'neg_log_loss':
        # score's sign inverted [log_loss in [0, 10**4)] 
        best_score = 10**4
        best_score_dev = 10**4
        best_cv_results = best_score*np.ones(n_splits)
        best_model_name = 'Worst'
    elif scoring == 'brier_score_loss':
        # score's for evaluation in [0, 1] 
        best_score = 1
        best_score_dev = 1
        best_cv_results = best_score*np.ones(n_splits)
        best_model_name = 'Worst'

    best_atts = best_score, best_score_dev, best_cv_results, best_model_name
    return best_atts


def classic_cv_model_evaluation(
        dx_train, dy_train, models_and_parameters, scoring, outer_cv,
        average_scores_across_outer_folds, scores_of_best_model, results,
        names, random_state):
    """Non-nested cross validation for model evaluation."""
    if (isinstance(scoring, list) or isinstance(scoring, dict)
            or isinstance(scoring, tuple)):
        raise TypeError("""'classic_cv_model_evaluation' method
        takes only single-metric score values.""")
    if scoring not in (None, 'accuracy', 'roc_auc', 'neg_log_loss'):
        raise ValueError("""%s is not a valid scoring value for method
            'classic_cv_model_evaluation'. Valid options are
            ['accuracy', 'roc_auc', 'neg_log_loss']""" % scoring)

    print("=== [task] Model evaluation with classic cross validation")
    print()

    wtr = calculate_sample_weight(dy_train)

    print("=== 'sample_weight'")
    print(wtr[0:5])
    print()

    for name, model in models_and_parameters.items():

        average_scores_across_outer_folds, scores_of_best_model = \
            single_classic_cv_evaluation(
                dx_train, dy_train, name, model, wtr, scoring, outer_cv,
                average_scores_across_outer_folds, scores_of_best_model,
                results, names, random_state)

    return average_scores_across_outer_folds, scores_of_best_model


def single_classic_cv_evaluation(
        dx_train, dy_train, name, model, sample_weight, scoring, outer_cv,
        average_scores_across_outer_folds, scores_of_best_model, results,
        names, random_state):
    """Non nested cross validation of single model."""
    if (isinstance(scoring, list) or isinstance(scoring, dict)
            or isinstance(scoring, tuple)):
        raise TypeError("""'single_classic_cv_evaluation' method takes only
        single-metric score values.""")
    if scoring not in (None, 'accuracy', 'roc_auc', 'neg_log_loss'):
        raise ValueError("""
        %s is not a valid scoring value for method
        'single_classic_cv_evaluation'. Valid options are ['accuracy',
        'roc_auc', 'neg_log_loss']""" % scoring)

    print()
    print("******* Evaluating model '%s'" % name)
    print()

    best_score = scores_of_best_model[0]
    best_score_dev = scores_of_best_model[1]
    best_cv_results = scores_of_best_model[2]
    # best_log_loss = log_loss_score
    # best_brier_score = scores_of_best_model[2]
    best_exec_time = scores_of_best_model[3]
    best_model_name = scores_of_best_model[4][0]
    best_model_estim = scores_of_best_model[4][1]
    best_nn_build_fn = scores_of_best_model[4][2]

    print("Best model: '%s'. Best score: %1.3f (%1.3f)"
          % (best_model_name, best_score, best_score_dev))

    # Create a temporary folder to store the transformers of the pipeline
    cachedir = mkdtemp()

    steps = []
    # ...

    if name == 'SVMClf_2nd':
        if len(dy_train) > 10000:
            name = 'Bagging_SVMClf_2nd'
            print("*** SVC detected, evaluating model '%s'" % name)

            # model = SVC(C=.01, kernel='linear', probability=True,
            #             class_weight='balanced', random_state=random_state)
            model.set_params(kernel='linear')

            n_estimators = 10
            bagging = BaggingClassifier(
                model, max_samples=1.0/n_estimators, n_estimators=n_estimators,
                random_state=random_state)
            model = bagging
        else:

            pass
        print("model:", model)
    else:
        pass

    steps.append((name, model))
    # transformers.append((name, model))

    ppline = Pipeline(steps, memory=cachedir)

    cv_success = 0

    try:
        t0 = time()
        cv_results = cross_val_score(
            ppline, dx_train, dy_train, cv=outer_cv,
            n_jobs=-2,
            pre_dispatch='2*n_jobs',
            scoring=scoring)
        t1 = time()
    except AttributeError as ae:
        print(ae)
    except JoblibValueError as jve:
        print(jve)
    except OverflowError as oe:
        print(oe)
    except Exception as e:
        print("Exception:", e)
    else:
        cv_success = 1

        names.append(name)
        if scoring == 'neg_log_loss':
            cv_results = -1*cv_results
        results.append(cv_results)

        print("Outer CV to get scores: successful.")

        exec_time = (t1 - t0)

        print('Execution time: %.2fs w %s.' % (exec_time, name))
        print()

        stats_for_sht = calculate_stats_for_nhst(
            name, cv_results, scoring)

        score, score_dev = stats_for_sht[0], stats_for_sht[1]
        func = stats_for_sht[2]

        print("*** Score for %s [%s]: %1.3f (%1.3f)"
              % (name, scoring.strip('neg_'), score, score_dev))
        print()

        # statistical testing of classifiers

        stats = score, score_dev, best_score, best_score_dev

        # statistical_hypothesis_testing
        sht_scores_dicts = compare_models_performance(
            name, model, exec_time, best_model_name, best_score_dev, 
            stats, cv_results, best_cv_results, 
            average_scores_across_outer_folds, scores_of_best_model, func, 
            cv_style="classic", scoring=scoring, params=None, 
            random_state=random_state)

    finally:
        if cv_success:
            print("Yay! Evaluation of model '%s' done." % name)
        else:
            print("Sorry. Evaluation of model '%s' failed." % name)

            sht_scores_dicts = \
            average_scores_across_outer_folds, scores_of_best_model

    del ppline

    # delete the temporary cache before exiting
    rmtree(cachedir)

    print()

    # average_scores_across_outer_folds, scores_of_best_model
    return sht_scores_dicts


def nested_rscv_model_evaluation(
        dx_train, dy_train, models_and_parameters, scoring, n_iter, inner_cv,
        outer_cv, average_scores_across_outer_folds, scores_of_best_model,
        results, names, random_state):
    """Nested cross-validation for model evaluation."""
    if (isinstance(scoring, list) or isinstance(scoring, dict)
            or isinstance(scoring, tuple)):
        raise TypeError("""'nested_rscv_model_evaluation' method allows only
        to perform single-metric evaluation of given estimator.""")
    if scoring not in (None, 'accuracy', 'roc_auc', 'neg_log_loss'):
        raise ValueError("""
        %s is not a valid scoring value for method
        'nested_rscv_model_evaluation'. Valid options are ['accuracy',
        'roc_auc', 'neg_log_loss']""" % scoring)

    print("=== [task] Model evaluation with nested-rscv")
    print()

    wtr = calculate_sample_weight(dy_train)

    print("=== 'sample_weight'")
    print(wtr[0:5])
    print()

    for name, (model, params) in models_and_parameters.items():

        average_scores_across_outer_folds, scores_of_best_model = \
            single_nested_rscv_evaluation(
                dx_train, dy_train, name, model, params, wtr, scoring, n_iter,
                inner_cv, outer_cv, average_scores_across_outer_folds,
                scores_of_best_model, results, names, random_state)

    return average_scores_across_outer_folds, scores_of_best_model


def single_nested_rscv_evaluation(
        dx_train, dy_train, name, model, params, sample_weight, scoring,
        n_iter, inner_cv, outer_cv, average_scores_across_outer_folds,
        scores_of_best_model, results, names, random_state, cv_meth='rscv'):
    """Non-nested cv for evaluation of a single model."""
    if (isinstance(scoring, list) or isinstance(scoring, dict)
            or isinstance(scoring, tuple)):
        raise TypeError("""'single_nested_rscv_evaluation' method takes only
        single-metric score values.""")
    if scoring not in (None, 'accuracy', 'roc_auc', 'neg_log_loss'):
        raise ValueError("""%s is not a valid scoring value for method
        'single_nested_rscv_evaluation'. Valid options are ['accuracy',
        'roc_auc', 'neg_log_loss']""" % scoring)

    print()
    print("******* Evaluating model '%s'" % name)
    print()

    best_score = scores_of_best_model[0]
    best_score_dev = scores_of_best_model[1]
    best_cv_results = scores_of_best_model[2]
    # best_log_loss = log_loss_score
    # best_brier_score = scores_of_best_model[2]
    best_exec_time = scores_of_best_model[3]
    best_model_name = scores_of_best_model[4][0]
    best_model_estim = scores_of_best_model[4][1]
    best_nn_build_fn = scores_of_best_model[4][2]

    print("Best model: '%s'. Best score: %1.3f (%1.3f)"
          % (best_model_name, best_score, best_score_dev))

    if name != 'DummyClf_2nd':

        steps = []
        # ...

        if name == 'SVMClf_2nd':
            name = 'Bagging_SVMClf_2nd'
            print("*** SVC detected, evaluating model '%s'" % name)

            # n_estimators = 10
            # SVC(kernel='linear', probability=True, class_weight='balanced',
            #     random_state=random_state)
            model.set_params(kernel='linear')

            n_estimators = 10
            bagging = BaggingClassifier(
                model, max_samples=1.0/n_estimators, n_estimators=n_estimators,
                random_state=random_state)

            model = bagging

            params = {
                name + '__' +
                k: v for k, v in pgd.Bagging_param_grid.items()}

            print("model:", model)

        else:
            pass

        steps.append((name, model))
        pipeline = Pipeline(steps)

        # estimate score [accuracy, roc_auc, ...] on
        # the stratified k-fold splits of the data
        if cv_meth == 'rscv':
            clf = RandomizedSearchCV(
                pipeline, param_distributions=params, iid=False, cv=inner_cv, 
                n_iter=n_iter, scoring=scoring, random_state=random_state)
        elif cv_meth == 'bscv':
            # check 'params' is of type BayesSearchCV's search_spaces
            clf = BayesSearchCV(
                pipeline, search_spaces=params, iid=False, cv=inner_cv, 
                n_iter=n_iter, scoring=scoring, random_state=random_state)
        else:
            raise ValueError("%s is not a valid value for var 'cv_meth', "
                             "valid values are ['rscv', 'bscv']")
    else:

        clf = model

    cv_success = 0
    try:
        t0 = time()
        cv_results = cross_val_score(
            clf, dx_train, dy_train, cv=outer_cv, n_jobs=-2, pre_dispatch=9,
            scoring=scoring)
        t1 = time()
    except AttributeError as ae:
        print(ae)
    except JoblibValueError as jve:
        print(jve)
    except OverflowError as oe:
        print(oe)
    except Exception as e:
        print("Exception:", e)
    else:
        cv_success = 1

        names.append(name)
        if scoring == 'neg_log_loss':
            cv_results = -1*cv_results
        results.append(cv_results)

        print("Outer CV to get scores: successful.")

        exec_time = (t1 - t0)

        print('Execution time: %.2fs w %s.' % (exec_time, name))
        print()

        stats_for_sht = calculate_stats_for_nhst(
            name, cv_results, scoring)

        score, score_dev = stats_for_sht[0], stats_for_sht[1]
        func = stats_for_sht[2]

        print("*** Score for %s [%s]: %1.3f (%1.3f)"
              % (name, scoring.strip('neg_'), score, score_dev))
        print()

        # statistical testing of classifiers

        stats = score, score_dev, best_score, best_score_dev

        # statistical_hypothesis_testing
        # compare_models_performance
        sht_scores_dicts = compare_models_performance(
            name, model, exec_time, best_model_name, best_score_dev, 
            stats, cv_results, best_cv_results, 
            average_scores_across_outer_folds, scores_of_best_model, func, 
            cv_style="xscv", scoring=scoring, params=params, 
            random_state=random_state)

    finally:
        if cv_success:
            print("Yay! Evaluation of model '%s' done." % name)
        else:
            print("Sorry. Evaluation of model '%s' failed." % name)

            sht_scores_dicts = \
            average_scores_across_outer_folds, scores_of_best_model

    print()

    # returns average_scores_across_outer_folds, scores_of_best_model
    return sht_scores_dicts 


# Before this, you should select between 'interactive' or 'auto' mode

# if interactive, you can ask user for a learning mode

def learning_mode(df):

    df_length = len(df.index)
    df_size = df.shape[1]*df.shape[0]

    assert df_length == df.shape[0]
    
    msg = "Are you in a hurry?"
    mood = au.say_yes_or_no(msg, getmood=True)

    if mood in {"YES", "yes", "Y", "y"}:
        # random 10% of dataframe

        learn = 'quick'
        print("You're in a hurry, let's speed things up.")
        print("We'll do a quick evaluation using lightly pre-optimized models.")

        # 10 features and one target --> 11 columns
        if df_length > 10000 and df_size > 110000:
            print("Dataframe length = %d > 10000" % df_length)
            print("Dataframe size = %d > 110000" % df_size)
            print("Reducing dataframe length to 10000 to speed things up.")

            frac = 10000/df_length
            print("frac: %1.3f" % frac)
            df = df.sample(frac=frac)   # frac=0.1
            # or dataframe.iloc[[indexes],:]
            # hurry_mode=1
            df_size = df.shape[1]*df.shape[0]
            print("After reduction, df. size = ", df_size)
            print("Dataframe length = ", len(df.index))
            print()

    else:
        if df_length <= 10000 or df_size <= 110000:
            if df_length <= 10000:
                print("Dataframe length = %d <= 10000" % df_length)
            if df_size <= 110000:
                print("Dataframe size = %d <= 110000" % df_size)
            learn = 'standard'
        else:
            print("Dataframe length = %d" % df_length)
            print("Dataframe size = %d" % df_size)
            learn = 'quick'
            # learn = 'large'

    return learn, df
