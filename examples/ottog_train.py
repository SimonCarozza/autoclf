"""Learning from Otto Gropu sales data."""

from sklearn.pipeline import Pipeline
from pandas import read_csv
# from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np

import os

from autoclf.classification import eval_utils as eu
from autoclf.classification import evaluate as eva
import autoclf.getargs as ga
from pkg_resources import resource_string
from io import StringIO

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# starting program
if __name__ == '__main__':

    print("### Probability Calibration Experiment 'Otto Group' "
          "-- CalibratedClassifierCV with cv=cv (no prefit) ###")
    print()

    d_name = ga.get_name()

    if d_name is None:
        d_name = "OttoG"

    seed = 7
    np.random.seed(seed)

    # place your code here
    try:
        df = read_csv("datasets/otto_group_train.csv", delimiter=",")
    except FileNotFoundError as fe:
        ottog_bytes = resource_string(
            "autoclf", os.path.join("datasets", 'otto_group_train.csv'))
        ottog_file = StringIO(str(ottog_bytes,'utf-8'))

        df = read_csv(ottog_file, delimiter=",")
    except Exception as e:
        raise e
    
    df_length = len(df.index)

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

    print()

    eva.perform_classic_cv_evaluation_and_calibration(
        auto_feat_eng_data, scoring, Y_type, labels, d_name, seed)

    input("=== [End Of Program] Enter key to continue... \n")
