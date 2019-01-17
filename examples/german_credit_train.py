from pandas import read_csv
# from pandas.plotting import scatter_matrix
import numpy as np
import os

from autoclf.classification import eval_utils as eu
from autoclf.classification import evaluate as eva
import autoclf.getargs as ga
from pkg_resources import resource_string
from io import StringIO
import urllib


# starting program
if __name__ == '__main__':

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

    try:
        df = read_csv(
            "datasets/german-credit.csv", header=None, delimiter=" ",
            names=names)
    except FileNotFoundError as fe:
        german_bytes = resource_string(
            "autoclf", os.path.join("datasets", 'german-credit.csv'))
        german_file = StringIO(str(german_bytes,'utf-8'))

        df = read_csv(german_file, header=None, delimiter=" ",
        names=names)
    except Exception as e:
        raise e

    print(df.shape)

    description = df.describe()
    print("Description - no encoding:\n", description)

    print()
    # input("Enter key to continue... \n")

    # check dataframe size, select learning strategy accordingly

    target = 'status'

    # target = 1 if target == 2 else 0
    df[target] = np.where(df[target] == 2, 1, 0)

    learnm, ldf = eu.learning_mode(df)

    odf = None
    if len(ldf.index) < len(df.index):

        odf = df
        # dataframe have been reduced based on its size
        df = ldf

    print("Learning mode:", learnm)
    print()

    # auto_feat_eng_data, scoring, Y_type, classes = eu.split_and_X_encode(
    #     df, target, 0.2, random_state=seed)

    sltt = eu.scoring_and_tt_split(df, target, 0.2, seed)

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

    print()
    print("scoring:", scoring)
    print()

    auto_feat_eng_data = eu.auto_X_encoding(sltt, seed)

    print()

    eva.select_evaluation_strategy(
        auto_feat_eng_data, target, 0.2, odf, scoring, Y_type, labels=classes,
        d_name=d_name, random_state=seed, learn=learnm)

    input("=== [End Of Program] Enter key to continue... \n")
