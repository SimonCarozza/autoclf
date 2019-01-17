"""Hyperparameter grids and distros for GridSearchCV and RandomizedSearchCV."""

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

xgb_import = 0
try:
    from xgboost import XGBClassifier
except ImportError as ie:
    print(ie)
else:
    xgb_import = 1

from scipy.stats import expon as sp_exp
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_unif
from scipy.stats import beta as sp_beta

import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# list of candidate hyperparameter grids for GSCV and distros for RSCV
# param spaces for searching with Hyperopt

#

LogR_gscv_param_grid = dict(
     LogRClf_2nd__C=[0.01, 0.1, 1., 10., 100.],
     LogRClf_2nd__fit_intercept=[True, False],
     LogRClf_2nd__class_weight=[None, 'balanced'],
     LogRClf_2nd__max_iter=[10, 50, 100, 250, 500, 1000]
     )

LogR_param_grid = dict(
     LogRClf_2nd__C=sp_exp(scale=100),
     LogRClf_2nd__fit_intercept=[True, False],
     LogRClf_2nd__class_weight=[None, 'balanced'],
     LogRClf_2nd__max_iter=sp_randint(10, 1e5)
     )


# BernoulliNB

BernNB__gscv_param_grid = dict(
     BernNBClf_2nd__alpha=[.001, .01, .1, 1., 10., 100.],
     BernNBClf_2nd__fit_prior=[False, True]
     )


BernNB_param_grid = dict(
     BernNBClf_2nd__alpha=sp_exp(scale=100),
     BernNBClf_2nd__fit_prior=[False, True]
     )


# GaussianNB only one paramater: priors=None

# no parameters


# LDAClf_2nd -- you may perform dimensionality reduction directly w LDAClf_2nd

LDA_gscv_param_grid = dict(
    LDAClf_2nd__solver=['lsqr', 'eigen'],
    LDAClf_2nd__tol=[0.0001, 0.001, 0.01, 0.1, 1.],
    LDAClf_2nd__shrinkage=[None, 'auto']
    )

LDA_param_grid = dict(
    LDAClf_2nd__solver=['lsqr', 'eigen'],
    LDAClf_2nd__tol=sp_exp(scale=1),
    LDAClf_2nd__shrinkage=[None, 'auto']
    )


# QDA

QDA_gscv_param_grid = dict(
    QDAClf_2nd__tol=[0.0001, 0.001, 0.01, 0.1, 1.]
    )

QDA_param_grid = dict(
    QDAClf_2nd__tol=sp_exp(scale=1)
    )


# KNeighborsClf_2nd

KNC_gscv_param_grid = dict(
    KNeighborsClf_2nd__weights=['uniform', 'distance'],
    KNeighborsClf_2nd__n_neighbors=[1, 5, 10, 30, 50]
    )

KNC_param_grid = dict(
    KNeighborsClf_2nd__weights=['uniform', 'distance'],
    KNeighborsClf_2nd__n_neighbors=sp_randint(1, 50)
    )


# Decision Trees

DTC_gscv_param_grid = dict(
    DecisionTreeClf_2nd__max_depth=[None, 3, 5, 10, 15],
    DecisionTreeClf_2nd__min_samples_split=np.arange(2, 31, 3),
    DecisionTreeClf_2nd__min_samples_leaf=np.arange(5, 111, 10)
    )

DTC_param_grid = dict(
    DecisionTreeClf_2nd__max_depth=sp_randint(3, 15),
    DecisionTreeClf_2nd__min_samples_split=sp_randint(2, 30),
    DecisionTreeClf_2nd__min_samples_leaf=sp_randint(5, 150)
    )


# RandomForestClf_2nd

RFC_gscv_param_grid = dict(
    RandomForestClf_2nd__n_estimators=[10, 30, 50, 100, 200, 500, 1000],
    RandomForestClf_2nd__criterion=['gini', 'entropy'],
    RandomForestClf_2nd__max_features=[0.25, 'auto', 'sqrt'],
    RandomForestClf_2nd__class_weight=[None, 'balanced', 'balanced_subsample']
    )

RFC_param_grid = RFC_gscv_param_grid
RFC_param_grid['RandomForestClf_2nd__n_estimators'] = sp_randint(100, 1000)


# ExtraTrees Clf

ETC_param_grid = dict(
    ExtraTreesClf_2nd__n_estimators=RFC_param_grid[
        'RandomForestClf_2nd__n_estimators'],
    # ExtraTreesClf_2nd__n_estimators=sp_randint(100, 5000),
    ExtraTreesClf_2nd__criterion=['gini', 'entropy'],
    ExtraTreesClf_2nd__max_features=[0.25, 'auto', 'sqrt'],
    ExtraTreesClf_2nd__class_weight=[None, 'balanced', 'balanced_subsample']
    )


# GradientBoosting Clf

GBC_gscv_param_grid = dict(
    # GBoostingClf_2nd__loss=['deviance', 'exponential'],
    GBoostingClf_2nd__learning_rate=[.0001, .001, .01, .1, 10.],
    GBoostingClf_2nd__n_estimators=[100, 200, 500, 1000],
    GBoostingClf_2nd__criterion=['friedman_mse', 'mse', 'mae'],
    GBoostingClf_2nd__max_features=[None, 'auto', 'sqrt', 'log2'],
    GBoostingClf_2nd__max_depth=[3, 5, 10, 25, 50],
    )

GBC_param_grid = dict(
    # GBoostingClf_2nd__loss=['deviance', 'exponential'],
    GBoostingClf_2nd__learning_rate=sp_beta(3, 1),  # sp_exp(10),
    GBoostingClf_2nd__n_estimators=sp_randint(100, 1000),
    GBoostingClf_2nd__criterion=GBC_gscv_param_grid[
        'GBoostingClf_2nd__criterion'],
    GBoostingClf_2nd__max_features=GBC_gscv_param_grid[
        'GBoostingClf_2nd__max_features'],
    GBoostingClf_2nd__max_depth=sp_randint(5, 50),
    )


# SVC

SVC_gscv_param_grid = dict(
    SVMClf_2nd__C=[0.01, .1, 1, 10, 100, 1000],
    SVMClf_2nd__kernel=['linear', 'poly', 'rbf', 'sigmoid'],
    SVMClf_2nd__degree=np.arange(1, 5),
    SVMClf_2nd__gamma=['auto', 0.001, 0.01, 0.1, 1., 10., 100.],
    SVMClf_2nd__coef0=[0., 1., 5., 10., 20., 50.],
    SVMClf_2nd__class_weight=[None, 'balanced'])

SVC_param_grid = dict(
    SVMClf_2nd__C=sp_exp(1000),
    SVMClf_2nd__kernel=['linear', 'poly', 'rbf', 'sigmoid'],
    SVMClf_2nd__degree=sp_randint(1, 5),
    SVMClf_2nd__gamma=sp_exp(100),    # ['auto', sp_exp(100)]
    SVMClf_2nd__coef0=sp_exp(scale=50),
    SVMClf_2nd__class_weight=[None, 'balanced'])


# LinearSVC

LinSVC_gscv_param_grid = dict(
    # LinearSVMClf_2nd__penalty=['l1', 'l2'],
    LinearSVMClf_2nd__C=[0.01, 0.1, 1, 5, 10, 100],
    # LinearSVMClf_2nd__dual=[False, True],
    LinearSVMClf_2nd__tol=[0.0001, 0.001, 0.01, 0.1, 1],
    LinearSVMClf_2nd__class_weight=[None, 'balanced'])

LinSVC_param_grid = dict(
    # linSVC__penalty=['l1', 'l2'],
    LinearSVMClf_2nd__C=sp_exp(scale=100),
    LinearSVMClf_2nd__tol=sp_exp(scale=1),
    # LinearSVMClf_2nd__dual=[False, True],
    LinearSVMClf_2nd__class_weight=[None, 'balanced']
    )


one_to_left = sp_beta(3, 1)   # sp_beta(10, 1)
from_zero_positive = sp_exp(0, 50)
second_half = sp_unif(0.5, 1-0.5)

# ADABoost

AdaBC_param_grid = dict(
    AdaBClf_2nd__n_estimators=sp_randint(100, 1000),
    AdaBClf_2nd__learning_rate=one_to_left,
    )

# XGBoost

if xgb_import:
    # XGBClf_2nd__colsample_bytree=one_to_left,
    # XGBClf_2nd__subsample=one_to_left,
    XGBC_param_grid = dict(
        XGBClf_2nd__n_estimators=sp_randint(100, 1000),
        XGBClf_2nd__max_depth=sp_randint(3, 40),
        XGBClf_2nd__learning_rate=one_to_left,
        XGBClf_2nd__gamma=sp_beta(0.5, 1),
        XGBClf_2nd__reg_alpha=from_zero_positive,
        XGBClf_2nd__min_child_weight=from_zero_positive,
        XGBClf_2nd__subsample=second_half,
        # XGBClf_2nd__colsample_bytree=second_half
        )

# [[3,2,1], [3,1,2], [2,3,1], [2,1,3], [1,2,3], [1,3,2]; [3,1,1], [5, 1, 1]]
VC_3_param_grid = dict(
    weights=[None, [2, 1, 1], [1, 2, 1], [1, 1, 2],
                   [3, 2, 1], [3, 1, 2], [2, 3, 1],
                   [2, 1, 3], [1, 2, 3], [1, 3, 2],
                   [3, 1, 1], [1, 3, 1], [1, 1, 3],
                   [5, 1, 1], [1, 5, 1], [1, 1, 5]],
    flatten_transform=[None, True]
             )

Bagging_gscv_param_grid = dict(
    n_estimators=[3, 5, 10, 15, 30], max_samples=[0.1, 0.2, 0.3, 0.5, 1.0])

Bagging_param_grid = dict(
    n_estimators=[3, 5, 10, 15, 30], max_samples=sp_unif(scale=1))


###

# dict of models and their associated parameters
# if it comes out that the best model is LogReg, no comparison is needed

seed = 7
# np.random.seed(seed)


# 'DummyClf_2nd' initialized inside classification_functions.py at run-time

# sklearn 0.20.1 default {solver='lbfgs', multi_class='auto'}
# sklearn 0.19.1 {solver='liblinear', multi_class='ovr'}
full_search_models_and_parameters = {
    'LogRClf_2nd': (LogisticRegression(random_state=seed), 
        LogR_param_grid),
    'BernNBClf_2nd': (BernoulliNB(), BernNB_param_grid),
    'GaussianNBClf_2nd': (GaussianNB(), dict()),
    'LDAClf_2nd': (LinearDiscriminantAnalysis(), LDA_param_grid),
    'QDAClf_2nd': (QuadraticDiscriminantAnalysis(), QDA_param_grid),
    'KNeighborsClf_2nd': (KNeighborsClassifier(), KNC_param_grid),
    'DecisionTreeClf_2nd': (
        DecisionTreeClassifier(
            class_weight='balanced', random_state=seed), DTC_param_grid),
    'RandomForestClf_2nd': (
        RandomForestClassifier(
            oob_score=True, random_state=seed), RFC_param_grid),
    'ExtraTreesClf_2nd': (
        ExtraTreesClassifier(
            oob_score=True, bootstrap=True, random_state=seed),
            ETC_param_grid),
    'GBoostingClf_2nd': (
        GradientBoostingClassifier(random_state=seed), GBC_param_grid),
    'AdaBClf_2nd': (AdaBoostClassifier(random_state=seed), AdaBC_param_grid),
    # 'XGBClf_2nd': (XGBClassifier(seed=seed), XGBC_param_grid),
    'LinearSVMClf_2nd': (LinearSVC(dual=False, max_iter=1e4), LinSVC_param_grid),
    'SVMClf_2nd': (
        SVC(kernel='rbf', probability=True, class_weight='balanced'),
        SVC_param_grid)
}

if xgb_import:
    full_search_models_and_parameters['XGBClf_2nd'] =\
    (XGBClassifier(seed=seed), XGBC_param_grid)


# Lightly pre-optimized models to be used a starting point for ML problems
# R. Olson et al.: https://arxiv.org/abs/1708.05070

# sklearn 0.20.1 default {solver='lbfgs', multi_class='auto'}
# sklearn 0.19.1 {solver='liblinear', multi_class='ovr'}
starting_point_models_and_params = {
    'LogRClf_2nd': LogisticRegression(
        C=1.5, solver='liblinear', penalty='l1', fit_intercept=True, 
        class_weight='balanced', multi_class='ovr', 
        random_state=seed),
    'RandomForestClf_2nd': RandomForestClassifier(
        n_estimators=500, max_features=.25, criterion='entropy',
        oob_score=True,
        class_weight='balanced', random_state=seed),
    'ExtraTreesClf_2nd': ExtraTreesClassifier(
        n_estimators=1000, max_features='log2', criterion='entropy',
        class_weight='balanced', random_state=seed),
    'GBoostingClf_2nd': GradientBoostingClassifier(
        loss='deviance', n_estimators=500,  max_features='log2',
        random_state=seed),
    'SVMClf_2nd': SVC(C=.01, gamma=.1, kernel='poly', degree=3, coef0=10.,
                      probability=True, class_weight='balanced',
                      random_state=seed),
    }
