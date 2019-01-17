"""Prediction methods."""

import re
from numpy import amax, int8, int32, int64, ndarray
from operator import itemgetter
from sklearn.calibration import CalibratedClassifierCV


def set_prediction_label(pred, prob, labels):
    """
    Set prediction labels based on models output and ml problem

    ---
    pred: prediction

    prob: class probability

    labels: labels of target classes

    --- returns

    (pred, prob)
    """
    if labels is None:
        print("'NoneType' object.")
        raise TypeError("Argument 'labels' should be a tuple of strings")
    print(f"Prediction: {pred}, type of pred.:", type(pred))
    # print("length of proba:", len(prob))
    if len(prob) >= 2:
        # prob == 2: binary problem
        # prob > 2: multiclass problem
        # in multiclass: probability of most likely event to occur
        prob = amax(prob)
        if isinstance(pred, (int8, int32, int64)):
            for dx, val in list(enumerate(labels)):
                if pred == dx:
                    pred = val
        elif isinstance(pred, str):
            try:
                int(pred)
            except ValueError:
                pass
            except Exception:
                raise
            else:
                for dx, val in list(enumerate(labels)):
                    if pred == dx:
                        pred = val
        elif isinstance(pred, ndarray):
            index, value = max(enumerate(pred), key=itemgetter(1))
            pred = labels[index]
    else:
        # dealing with KerasClf's response to binary problem
        if prob < .5:
            pred = labels[0]
            prob = 1. - prob
        else:
            pred = labels[1]
    return pred, prob


def prediction_with_single_estimator(
        name, estimator, raw_X, data_X, pck_indexes, dx_indexes,
        labels, feature_str):
    """
    Make predictions using single estimator.

    ---
    name: estimator name

    estimator: sklearn pipeline made of multiple named steps

    raw_X: new non-preprocessed data 

    data_X: new data stripped of target variable

    pck_indexes: indexes to randomly pick new data for predictions

    dx_indexes: indexes of best features

    labels: new data labels

    feature_str: string with features to format (python >= 3.6 only) 
    """
    print()
    print("=== Predictor: '%s'" % name)
    print()
    predictions = estimator.predict(data_X)
    probas = estimator.predict_proba(data_X)
    raw_X_sample = []
    rnd_predictions = []
    rnd_probas = []
    for pi in pck_indexes:
        # print()
        # print(f"Probas[{pi}]:", probas[pi])
        raw_X_sample.append(raw_X[pi])
        rnd_predictions.append(predictions[pi])
        rnd_probas.append(probas[pi])
    rnd_preds_and_probas = zip(raw_X_sample, rnd_predictions, rnd_probas)
    for x, pred, prob in rnd_preds_and_probas:
        pred, prob = set_prediction_label(pred, prob, labels)
        X = (x[i] for i in dx_indexes)
        print(feature_str.format(*X))
        print("Result: '%s' (%.2f%%)" % (pred, prob*100))
    print()

# old method, it's here for comparison purpose
#
# def prediction_with_single_estimator(
#         name, estimator, raw_X, data_X, pck_indexes, dx_indexes,
#         bin_event, feature_str):
#     """Make predictions using single estimator."""
#     print()
#     print("=== Predictor: '%s'" % name)
#     predictions = estimator.predict(data_X)
#     probas = estimator.predict_proba(data_X)
#     raw_X_sample = []
#     rnd_predictions = []
#     rnd_probas = []
#     for pi in pck_indexes:
#         raw_X_sample.append(raw_X[pi])
#         rnd_predictions.append(predictions[pi])
#         rnd_probas.append(probas[pi])
#     rnd_preds_and_probas = zip(raw_X_sample, rnd_predictions, rnd_probas)
#     doom = ''
#     chance = 0.0
#     for x, pred, prob in rnd_preds_and_probas:
#         if len(prob) == 2:
#             # reference probability: that of positive event 'good'
#             chance = prob[1]
#             if pred == 0:
#                 doom = bin_event[0]
#             else:
#                 doom = bin_event[1]
#         else:
#             # dealing with KerasClf
#             chance = prob
#             if prob < .5:
#                 doom = bin_event[0]
#             else:
#                 doom = bin_event[1]
#         X = (x[i] for i in dx_indexes)
#         print(feature_str.format(*X))
#         print("Status: '%s' (%.2f%%)" % (doom, chance*100))
#     print()


def show_estimator_steps(estim):
    """Show estimator's steps."""
    name = ''
    if hasattr(estim, 'steps'):
        for step in estim.steps:
            print("step:", step)
            print()
            m = re.search("Clf", step[0])
            if m:
                name = step[0]
                if name == 'VClf_3_2nd':
                    print("%s's estimators:" % name, step[1].estimators_)
    elif isinstance(estim, CalibratedClassifierCV):
        steps = estim.get_params()["base_estimator__steps"]
        for step in steps:
            print("step:", step)
            print()
            m = re.search("Clf", step[0])
            if m:
                name = step[0]
    return name


def predictions_with_full_estimators(
        estimators, raw_X, data_X, pck_indexes, dx_indexes,
        labels, feature_str):
    """
    Use list of estimators to make predictions upon randomly picked new data.

    Estimators are trained pipelines with a classifier as the last step
    ---
    estimators: list of available (best) estimators from train_calibrate.py

    raw_X: new non-preprocessed data 

    data_X: new data stripped of target variable

    pck_indexes: indexes to randomly pick new data for predictions

    dx_indexes: indexes of best features

    labels: new data labels

    feature_str: string with features to format (python >= 3.6 only) 
    """
    for estim in estimators:
        print()
        print("Estimator:\n", estim)
        print()
        name = show_estimator_steps(estim)

        predicted = 0
        try:
            prediction_with_single_estimator(
                name, estim, raw_X, data_X, pck_indexes, dx_indexes,
                labels, feature_str)
        except NameError as ne:
            print(ne)
        except TypeError as te:
            print(te)
        except ValueError as ve:
            print(ve)
        except Exception as e:
            print(e)
        else:
            predicted = 1
        finally:
            if predicted:
                print("Prediction with '%s' successful." % name)
            else:
                print("Prediction with '%s' failed." % name)
        print()
