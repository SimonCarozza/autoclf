# from pandas.plotting import scatter_matrix
from pandas import DataFrame
import numpy as np

import sys

import autoclf.classification.eval_utils as eu
import autoclf.classification.evaluate as eva
import autoclf.getargs as ga
from autoclf import auto_utils as au
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# starting program
if __name__ == '__main__':

    plt.style.use('ggplot')

    print()
    print("### Probability Calibration Experiment -- CalibratedClassifierCV "
          "with cv=cv (no prefit) ###")
    print()

    d_name = ga.get_name()

    if d_name is None:
        d_name = "Iris"

    seed = 7
    np.random.seed(seed)

    # load iris data
    iris = load_iris()

    f_names = [name.strip(' (cm)') for name in iris.feature_names]

    df = DataFrame(data=iris.data, columns=f_names)

    target = 'class'

    df_target = DataFrame(data=iris.target, columns=[target])

    # complete dataframe: features + target

    df[target] = df_target

    print(df.shape)

    print("Dataframe description - no encoding:\n", df.describe())
    print()
    print()
    print("=== [task] Train-test split + early pre-processing.")
    print()

    # Missing Attribute Values: None

    auto_feat_eng_data, scoring, Y_type, labels = eu.split_and_X_encode(
        df, target, 0.2, random_state=seed)

    print()
    print("scoring:", scoring)
    # input("Enter key to continue... \n")

    print()
    print()

    eva.perform_nested_cv_evaluation_and_calibration(
        auto_feat_eng_data, scoring, Y_type, labels=labels, d_name=d_name, 
        random_state=seed)

    # learnm = 'quick'

    # eva.select_evaluation_strategy(
    #     auto_feat_eng_data, target, 0.2, None, scoring, Y_type, labels=labels,
    #     d_name=d_name, random_state=seed, learn=learnm)

    input("=== [End Of Program] Enter key to continue... \n")

