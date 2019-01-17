from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

from pandas import DataFrame
import numpy as np
from sklearn.datasets import load_digits
import sys

from autoclf import auto_utils as au
from autoclf.classification import eval_utils as eu
from autoclf.classification import evaluate as eva
import autoclf.getargs as ga

import matplotlib.pyplot as plt


# starting program
if __name__ == '__main__':

    plt.style.use('ggplot')

    print()
    print("### Probability Calibration Experiment -- CalibratedClassifierCV "
          "with cv=cv (no prefit) ###")
    print()

    print(ga.get_n_epoch())

    d_name = ga.get_name()

    if d_name is None:
        d_name = 'Digits'

    seed = 7
    np.random.seed(seed)

    # load digits data
    digits = load_digits()

    df = DataFrame(data=digits.data)

    target = 'class'

    df_target = DataFrame(data=digits.target, columns=[target])

    # complete dataframe: features + target

    df[target] = df_target

    if isinstance(df, DataFrame):
        pass
    else:
        raise ValueError("df must be a pandas.DataFrame")

    print(df.shape)

    print("Dataframe description - no encoding:\n", df.describe())
    print()

    # check dataframe to estimate a learning mode

    learnm, ldf = eu.learning_mode(df)

    odf = None
    if len(ldf.index) < len(df.index):

        odf = df
        # dataframe have been reduced based on its size
        df = ldf

    print("Learning mode:", learnm)
    print()

    print()
    print("=== [task] Train-test split + early pre-processing.")
    print()

    # Missing Attribute Values: None

    auto_feat_eng_data, scoring, Y_type, labels = eu.split_and_X_encode(
        df, target, 0.2, random_state=seed)

    print()

    eva.select_evaluation_strategy(
        auto_feat_eng_data, target, 0.2, odf, scoring, Y_type, labels=labels,
        d_name=d_name, random_state=seed, learn=learnm)

    input("\n=== [End Of Program] Enter key to continue... \n")