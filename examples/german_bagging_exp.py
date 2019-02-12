from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier

from pandas import read_csv
# from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from autoclf.classification import eval_utils as eu
from autoclf.classification.evaluate import create_ensemble_of_best_models
from autoclf.classification import param_grids_distros as pgd
from autoclf import auto_utils as au
from autoclf.classification import train_calibrate as tc
import autoclf.getargs as ga
from pkg_resources import resource_string
from io import StringIO


# starting program
if __name__ == '__main__':

    plt.style.use('ggplot')

    print("### Probability Calibration Experiment -- CalibratedClassifierCV "
          "with cv=cv (no prefit) ###")
    print()

    d_name = ga.get_name()

    if d_name is None:
        d_name = "GermanCr"

    seed = 7
    np.random.seed(seed)

    names = ['checkin_acc', 'duration', 'credit_history', 'purpose', 'amount',
             'saving_acc', 'present_emp_since', 'inst_rate', 'personal_status',
             'other_debtors', 'residing_since', 'property', 'age',
             'inst_plans', 'housing', 'num_credits', 'job', 'dependents',
             'telephone', 'foreign_worker', 'status']

    german_bytes = resource_string(
        "autoclf", os.path.join("datasets", 'german-credit.csv'))
    german_file = StringIO(str(german_bytes,'utf-8'))

    df = read_csv(german_file, header=None, delimiter=" ",
    names=names)

    print(df.shape)

    description = df.describe()
    print("Description - no encoding:\n", description)

    print()

    target = 'status'

    # target = 1 if target == 2 else 0
    df[target] = np.where(df[target] == 2, 1, 0)

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

    # Evaluation of best modelwith nested CV -- inner: RSCV

    best_score = 0.5
    best_score_dev = 0.5
    best_cv_results = np.zeros(n_splits)
    best_exec_time = 31536000    # one year in seconds
    best_model = ('Random', None, None)

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

    strategy = 'stratified'  # 'most_frequent'

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

    all_models_and_parameters = dict()

    # Let's assume the best model is GaussianNB 

    print("=== [task] Comparing DummyClassifier to ensemble of GaussianNBs")
    print()
    
    bag_estim, bag_estim_name, bag_param_grid = create_ensemble_of_best_models(
        'GaussianNBClf_2nd', GaussianNB(), seed)

    average_scores_and_best_scores = eu.single_nested_rscv_evaluation(
        X_train_transformed, y_train, bag_estim_name,
        bag_estim, bag_param_grid, wtr, scoring, n_iter,
        inner_cv, outer_cv, dict(), scores_of_best_model, results, 
        names, seed)

    all_models_and_parameters[bag_estim_name] = (
        bag_estim, bag_param_grid
        )

    print()
    au.box_plots_of_models_performance(results, names)

    print()
    print("=== After nested CV evaluation of ensemble of GaussianNBs...")
    print()

    scores_of_best_model = average_scores_and_best_scores[1]

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