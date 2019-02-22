"""Nested-cv to evaluate models and learn who'll survive the Titanic."""

from sklearn.pipeline import Pipeline
from pandas import read_csv
# from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np

import os

from autoclf.classification import eval_utils as eu
from autoclf.classification import evaluate as eva
from autoclf.classification import param_grids_distros as pgd
import autoclf.getargs as ga
from pkg_resources import resource_string
from io import StringIO

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# starting program
if __name__ == '__main__':

    print("### Probability Calibration Experiment -- CalibratedClassifierCV "
          "with cv=cv (no prefit) ###")
    print()

    d_name = ga.get_name()

    if d_name is None:
        d_name = "Titanic"

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # load data

    try:
        df = read_csv(
            'datasets/titanic_train.csv', delimiter=",",
            na_values={'Age': '', 'Cabin': '', 'Embarked': ''},
            dtype={'Name': 'category', 'Sex': 'category',
                   'Ticket': 'category', 'Cabin': 'category',
                   'Embarked': 'category'})
        print("Found data in 'autoclf/datasets'")
    except FileNotFoundError as fe:
        titanic_bytes = resource_string(
            "autoclf", os.path.join("datasets", 'titanic_train.csv'))
        titanic_file = StringIO(str(titanic_bytes,'utf-8'))

        names = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp',
                 'Parch','Ticket','Fare','Cabin','Embarked']

        df = read_csv(
            titanic_file, delimiter=",",
            # header=0, names=names,
            na_values={'Age': '', 'Cabin': '', 'Embarked': ''},
            dtype={'Name': 'category', 'Sex': 'category',
                   'Ticket': 'category', 'Cabin': 'category',
                   'Embarked': 'category'})
    except Exception as e:
        raise e


    # data exploration

    print("shape: ", df.shape)

    # statistical summary
    description = df.describe()
    print("description - no encoding:\n", description)
    print()

    plt.style.use('ggplot')

    # Feature-Feature Relationships
    # scatter_matrix(df)

    print()

    # too many missing values in 'Cabin' columns: about 3/4
    print("Dropping 'Cabin' column -- too many missing values")
    df.drop(['Cabin'], axis=1, inplace=True)

    print()
    print("Now, shape: ", df.shape)

    print("df.head():\n", df.head())
    print()

    description = df.describe()
    print("Once again, description - no encoding:\n", description)

    print()
    # input("Enter key to continue... \n")

    target = 'Survived'

    # check dataframe to estimate a learning mode

    learnm, ldf = eu.learning_mode(df)

    odf = None
    if len(ldf.index) < len(df.index):

        odf = df
        # dataframe have been reduced based on its size
        df = ldf

    print("Learning mode:", learnm)
    print()

    # feature engineering

    sltt = eu.scoring_and_tt_split(df, target, 0.2, seed)

    X_train, X_test, y_train, y_test = sltt['arrays']

    scoring = sltt['scoring']
    Y_type = sltt['target_type']
    labels = sltt['labels']

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

    print(y_train[:3])
    # input("Enter key to continue... \n")

    print()
    print("scoring:", scoring)
    print()

    auto_feat_eng_data = eu.auto_X_encoding(sltt, seed)

    print()

    eva.select_evaluation_strategy(
        auto_feat_eng_data, scoring, Y_type, labels=labels,
        d_name=d_name, random_state=seed, learn=learnm)

    input("=== [End Of Program] Enter key to continue... \n")
