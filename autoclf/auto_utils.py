from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

from sklearn.externals import joblib as jl
from sklearn.externals.joblib.my_exceptions import JoblibValueError
import errno
import os
import sys
from re import search
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as Gaussian
from sklearn.model_selection import learning_curve
import autoclf.getargs as ga


#

def select_X_transformer():
    # Scaling X

    # get int input in range [1, 5]
    choice = ga.get_scaler_name()
    scaler_tuple = ()

    if choice is None:
        is_valid = 0
        choice = 0

        msg = ("Select scaler:\n"
               "\t[1] standard, [2] robust, [3] minmax,\n"
               "\t[4] normalizer, [5] uniform quantile transf.,\n"
               "\t[6] gaussian quantile tr.\n")

        while not is_valid:
            try:
                choice = int(input(msg))
                if choice >= 1 and choice <= 5:
                    # set it to 1 to validate input and to terminate the while loop
                    is_valid = 1
                else:
                    print("Invalid number. Try again...")
            except ValueError as e:
                print("'%s' is not a valid integer." % e.args[0].split(": ")[1])

    # You don't need to scale data (X) a priori
    if choice == 1:
        print("Standard data scaling.")
        sscaler = StandardScaler()
        # X = sscaler.fit_transform(dX)
        scaler_tuple = ('Standardizer', sscaler)
    elif choice == 2:
        print("Robust data scaling: robust to outliers.")
        rscaler = RobustScaler()
        scaler_tuple = ('Robust_Scaler', rscaler)
    elif choice == 3:
        print("MinMax data scaling: alternative to standard scaling.")
        mmscaler = MinMaxScaler()
        scaler_tuple = ('MinMaxScaler', mmscaler)
    elif choice == 4:
        print("Data normalization. Useful for feeding neural networks.")
        normalizer = Normalizer()
        scaler_tuple = ('Normalizer', normalizer)
    # ! only for scikit-learn >= 0.19 !
    elif choice == 5:
        qtrans_unif = QuantileTransformer()
        scaler_tuple = ('Uniform_QuantileTransformer', qtrans_unif)
    else:
        qtrans_normal = QuantileTransformer(output_distribution='normal')
        scaler_tuple = ('Normal_QuantileTransformer', qtrans_normal)
    print("Pipeline evaluation with '" + scaler_tuple[0] + "'.")

    print()

    return scaler_tuple


def say_yes_or_no(msg, getmood=False):

    mood = None

    if getmood:
        mood = ga.get_mood()

    if mood is None:
        is_valid = 0
        choice = ""

        while not is_valid:
            choice = str(input(msg + "\n"))
            if choice in {"YES", "NO", "yes", "no", "Y", "y", "N", "n"}:
                is_valid = 1
            else:
                print("Invalid input. Write 'YES' or 'NO', please.")
    else:
        choice = mood

    print()

    return choice


def select_encoding_method():

    choice = ga.get_encoder_name()

    if choice is None:
        is_valid = 0
        choice = 0

        while not is_valid:
            try:
                choice = int(input("Select encoder: [1] Label-Encoding, [2] "
                                   "One-Hot-Encoding (computationally heavy)?\n"))
                if choice in (1, 2):
                    is_valid = 1
                else:
                    print("Invalid number. Try again...")
            except ValueError as e:
                print("'%s' is not a valid integer." % e.args[0].split(": ")[1])

        encoding = 'le' if choice == 1 else 'ohe'
    else:
        encoding = choice

    return encoding


# Performance methods -- More tweaks to improve performance

# save model

def save_model(model, filename, d_name=None):
    # curdir = os.path.dirname(__file__)
    curdir = os.getcwd()

    # directory = curdir + "\\models"
    directory = os.path.join(curdir, "models")

    try:
        os.makedirs(directory)
    except OSError as ose:
        if ose.errno != errno.EEXIST:
            raise

    if d_name is not None:
        filename = d_name + "_" + filename

    filename = os.path.join(directory, filename)

    try:
        with open(filename, 'wb') as f:
            jl.dump(model, f)
            print("File '%s' saved to disk." % filename)
    except OSError as ose:
        print(ose)
        print("Well darn.")


# save Keras models

def create_keras_model_filename(filename, d_name=None):

    # also, os.path.dirname(__file__)
    # curdir = os.path.dirname(os.path.abspath(__file__))
    curdir = os.getcwd()

    # directory = curdir + "\\models"
    directory = os.path.join(curdir, "models")

    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if d_name:
        filename = d_name + "_" + filename

    return os.path.join(directory, filename)


def check_search_space_of_params(nr_iter, param_grid):

    # n_iter = check_search_space_of_params(n_iter, param_grid)

    nr_of_params = 0
    for value in param_grid.values():
        for i in value:
            nr_of_params += 1

    if nr_of_params < nr_iter:
        nr_iter = nr_of_params

    return nr_iter


def input_choice(msg='', upper=20, task='split', choice=None):

    is_valid = 0

    if task == 'split':
        choice = ga.get_n_split()
        advice = "Lower the nr. of fold splits."
    elif task == 'iter':
        advice = "Lower the nr. of iterations."
    else:
        raise ValueError("'%s' is not a valid task value. Valid options are"
                         " ['split', 'iter']")

    while not is_valid:
        try:
            if choice is None:
                choice = int(input(msg))
                print()
        except ValueError as e:
            print("'%s' is not a valid integer." % e.args[0].split(": ")[1])
        else:
            if choice >= 1 and choice <= upper:
                is_valid = 1
            else:
                if choice > upper:
                    print("Evaluation is gonna take forever. " + advice)
                else:
                    print("Invalid number. Try again...")
                choice = None

    return choice


def select_nr_of_splits_for_kfold_cv():

    msg = "Specify the number of folds for you cross-validation strategy.\n"

    return input_choice(msg)


def select_nr_of_iterations(iter_obj='rscv', followup=False):

    n_iter = None
    msg = ''

    if iter_obj == 'rscv':
        upper = 50

        n_iter = ga.get_n_param_setting()

        if n_iter is None:
            msg = "Specify the number of parameter settings that are sampled by RSCV.\n"

    elif iter_obj == 'nn':
        upper = 1000

        # epochs: iterations on a dataset
        n_ep_list = ga.get_n_epoch()

        if n_ep_list is None:
            msg = "Specify the number of epochs for training of Keras NNss.\n"
            
        else:
            if followup:
                if len(n_ep_list) == 1:
                    # user provided one epoch value
                    msg = "Specify the number of epochs for training of Keras NNs.\n"
                else:
                    n_iter = n_ep_list[1]
                    msg = ''
            else:
                # no followup, one element in epochs list
                n_iter = n_ep_list[0]
                msg = ''
    else:
        raise ValueError("'%s' is not a valid iter_obj value. "
                         "Valid options are ['rscv', 'nn']")

    if n_iter is None:
        return input_choice(msg, upper, 'iter')
    else:
        return input_choice(msg, upper, 'iter', n_iter)


###

# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# Nonparametric Permutation Test
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from itertools import combinations
from math import factorial
try:
    from nose.tools import nottest
except ImportError:
    # Use a no-op decorator if nose is not available
    def nottest(f):
        return f

# decorator to prevent nose to consider
# this as a unit test due to "test" in the name
@nottest
def permutation_test(x, y, func='x_mean != y_mean', method='exact',
                     num_rounds=1024, seed=None):
    """
    Nonparametric permutation test.

    Parameters
    ----------
    x : list or numpy array with shape (n_datapoints,)
        A list or 1D numpy array of the first sample
        (e.g., the treatment group).
    y : list or numpy array with shape (n_datapoints,)
        A list or 1D numpy array of the second sample
        (e.g., the control group).
    func : custom function or str (default: 'x_mean != y_mean')
        function to compute the statistic for the permutation test.
        - If 'x_mean != y_mean', uses
          `func=lambda x, y: np.abs(np.mean(x) - np.mean(y)))`
           for a two-sided test.
        - If 'x_mean > y_mean', uses
          `func=lambda x, y: np.mean(x) - np.mean(y))`
           for a one-sided test.
        - If 'x_mean < y_mean', uses
          `func=lambda x, y: np.mean(y) - np.mean(x))`
           for a one-sided test.
    method : 'approximate' or 'exact' (default: 'exact')
        If 'exact' (default), all possible permutations are considered.
        If 'approximate' the number of drawn samples is
        given by `num_rounds`.
        Note that 'exact' is typically not feasible unless the dataset
        size is relatively small.
    num_rounds : int (default: 1024 == 2**10)
        The number of permutation samples if `method='approximate'`.
    seed : int or None (default: None)
        The random seed for generating permutation samples if
        `method='approximate'`.

    Returns
    -------
    p-value under the null hypothesis

    Examples
    --------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/

    """
    if method not in ('approximate', 'exact'):
        raise AttributeError('method must be "approximate"'
                             ' or "exact", got %s' % method)

    if isinstance(func, str):

        if func not in (
                'x_mean != y_mean', 'x_median != y_median',
                'x_mean > y_mean', 'x_mean < y_mean',
                'x_median > y_median', 'x_median < y_median'):
            raise AttributeError('Provide a custom function'
                                 ' lambda x,y: ... or a string'
                                 ' in ("x_mean != y_mean", '
                                 '"x_mean > y_mean", "x_mean < y_mean"'
                                 '"x_median > y_median", '
                                 '"x_median < y_median"')

        elif func == 'x_mean != y_mean':
            def func(x, y):
                return np.abs(np.mean(x) - np.mean(y))

        elif func == 'x_median != y_median':
            def func(x, y):
                return np.abs(np.median(x) - np.median(y))

        elif func == 'x_median > y_median':
            def func(x, y):
                return np.median(x) - np.median(y)

        elif func == 'x_median < y_median':
            def func(x, y):
                return np.median(y) - np.median(x)

        elif func == 'x_mean > y_mean':
            def func(x, y):
                return np.mean(x) - np.mean(y)

        else:
            def func(x, y):
                return np.mean(y) - np.mean(x)

    rng = np.random.RandomState(seed)

    m, n = len(x), len(y)
    combined = np.hstack((x, y))

    more_extreme = 0.
    reference_stat = func(x, y)

    # Note that whether we compute the combinations or permutations
    # does not affect the results, since the number of permutations
    # n_A specific objects in A and n_B specific objects in B is the
    # same for all combinations in x_1, ... x_{n_A} and
    # x_{n_{A+1}}, ... x_{n_A + n_B}
    # In other words, for any given number of combinations, we get
    # n_A! x n_B! times as many permutations; hoewever, the computed
    # value of those permutations that are merely re-arranged combinations
    # does not change. Hence, the result, since we divide by the number of
    # combinations or permutations is the same, the permutations simply have
    # "n_A! x n_B!" as a scaling factor in the numerator and denominator
    # and using combinations instead of permutations simply saves computational
    # time

    if method == 'exact':
        for indices_x in combinations(range(m + n), m):

            indices_y = [i for i in range(m + n) if i not in indices_x]
            diff = func(combined[list(indices_x)], combined[indices_y])

            if diff > reference_stat:
                more_extreme += 1.

        num_rounds = factorial(m + n) / (factorial(m)*factorial(n))

    else:
        for i in range(num_rounds):
            rng.shuffle(combined)
            if func(combined[:m], combined[m:]) > reference_stat:
                more_extreme += 1.

    return more_extreme / num_rounds


def box_plots_of_models_performance(results, names):
    fig = plt.figure()
    fig.suptitle('Model Comparison')
    # ax = fig.add_subplot(111)
    ax = fig.subplots(1, 1)
    plt.boxplot(results, meanline=True, showmeans=True)
    ax.set_xticklabels(names)
    plt.show()


def mad(a, c=Gaussian.ppf(3/4.), axis=0, center=np.median):
    """
    Compute the Median Absolute Deviation along given axis of an array.

    Parameters
    ----------
    a : array-like
        Input array.
    c : float, optional
        The normalization constant.  Defined as scipy.stats.norm.ppf(3/4.),
        which is approximately .6745.
    axis : int, optional
        The defaul is 0. Can also be None.
    center : callable or float
        If a callable is provided, such as the default `np.median` then it
        is expected to be called center(a). The axis argument will be applied
        via np.apply_over_axes. Otherwise, provide a float.

    Returns
    -------
    mad : float
        `mad` = median(abs(`a` - center))/`c`
    """
    a = np.asarray(a)
    if callable(center):
        center = np.apply_over_axes(center, a, axis)
    return np.median((np.abs(a-center))/c, axis=axis)


def plot_learning_curve(estimator, X, y, ylim=None, cv=None, scoring=None,
        n_jobs=1, serial=None, tuning='None', d_name=None,
        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if "y" is binary or multiclass,
        :class:'StratifiedKFold' used. If the estimator is not a classifier
        or if "y" is neither binary nor multiclass, :class:`KFold` is used.

        Refer 'User Guide'
        > http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
        for the various cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    serial : serial number of "estimator" to be distinguished from others.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that
        will be used to generate the learning curve. If the dtype is float,
        it is regarded as a fraction of the maximum size of the training set
        (that is determined by the selected validation method),
        i.e. it has to be within (0, 1]. Otherwise it is interpreted
        as absolute sizes of the training sets.
        Note that for classification the number of samples
        usually have to be big enough to contain at least one sample
        from each class. (default: np.linspace(0.1, 1.0, 5))
    """
    # curdir = os.path.dirname(__file__)
    curdir = os.getcwd()

    # directory = curdir + "\\results"
    directory = os.path.join(curdir, "results")

    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    name = ''
    for step in estimator.steps:
        m = search("Clf", step[0])
        if m:
            name = step[0]

    if d_name is not None:
        name = d_name + "_" + name

    clf_name = None

    if serial is None:
        try:
            search(r'(?<=_)\d{4}', name).group(0)
        except AttributeError as ae:
            print(ae)
            serial = "%04d" % randint(0, 1000)
        except Exception as e:
            print(e)
        else:
            serial = search(r'(?<=_)\d{4}', name).group(0)
            # clf_name = sys.argv[0][:5] + "_" + name + "_" + tuning + "_" + serial
            clf_name = name + "_" + tuning + "_" + serial

    if clf_name is None:
        # clf_name = sys.argv[0][:5] + "_" + name + "_" + tuning + "_" + serial
        clf_name = name + "_" + tuning + "_" + serial

    title = "Learning Curve for '%s'" % clf_name

    l_curve = 0

    try:
        train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
        train_sizes=train_sizes)
    except JoblibValueError as jve:
        print("Not able to complete learning process...")
    except RuntimeError as re:
        print(re)
    except ValueError as ve:
        print(ve)
    else:

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score: '%s'" % scoring)

        plt.grid()

        plt.fill_between(
            train_sizes, train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(
            train_sizes, test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(
            train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
        plt.plot(
            train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

        plt.legend(loc="best")

        l_curve = 1

        try:
            fig_name = os.path.join(
                curdir, directory, "learning_curve_" + clf_name)
        except FileNotFoundError as fe:
            print(fe)
        except Exception as e:
            print(e)

        saved = 0
    
        try:
            plt.savefig(fig_name + ".png", format='png')
            saved = 1
        except Exception as e:
            print(e)
        finally:
            if not saved:
                print("Sorry, could not save learning curve")
            else:
                print("Saved %s's learning curve" % name)

    finally:
        if not l_curve:
            print("Sorry. Learning Curve plotting failed.")
            print()
        else:
            plt.show()
