import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(os.getcwd()).parent))

import numpy as np
import graphviz
from scipy import stats
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.tree import export_graphviz
from sklearn.feature_selection import RFECV
from ML_tools.score_and_error import performance_scores


# PARAMETERS DISTRIBUTION FOR RANDOMSEARCHCV
param_dist = {'n_estimators': stats.randint(50, 500),
              'max_depth': stats.randint(1, 20)}


def RFPipeline_noPCA(df1, df2, n_iter, cv):
    """
    Creates pipeline that perform Random Forest classification on the data without Principal Component Analysis. The
    input data is split into training and test sets, then a Randomized Search (with cross-validation) is performed to
    find the best hyperparameters for the model.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Dataframe containing the features.
    df2 : pandas.DataFrame
        Dataframe containing the labels.
    n_iter : int
        Number of parameter settings that are sampled.
    cv : int
        Number of cross-validation folds to use.

    Returns
    -------
    pipeline_simple : sklearn.pipeline.Pipeline
        A fitted pipeline (includes hyperparameter optimization using RandomizedSearchCV and a Random Forest Classifier
        model).

    See Also
    --------
    RandomizedSearchCV : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    """

    X = df1.values
    y = df2.values
    region = list(df1.columns.values)

    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.2, random_state=7)

    pipeline_simple = Pipeline(steps=[("hyper_opt", RandomizedSearchCV(RandomForestClassifier(),
                                                                       param_distributions=param_dist,
                                                                       n_iter=n_iter,
                                                                       cv=cv,
                                                                       random_state=10))
                                      ]
                               )

    pipeline_simple.fit(X_tr, y_tr)

    y_pred = pipeline_simple.predict(X_tst)
    y_prob = pipeline_simple.predict_proba(X_tst)

    # SCORES WITH 95% CONFIDENCE INTERVAL
    scores = performance_scores(y_tst, y_pred, y_prob)

    for i in range(3):
        tree = pipeline_simple["hyper_opt"].best_estimator_[i]
        dot_data = export_graphviz(tree,
                                   feature_names=region,
                                   filled=True,
                                   impurity=False,
                                   proportion=True,
                                   class_names=["CN", "AD"])
        graph = graphviz.Source(dot_data)
        graph.render(view=True)

    return pipeline_simple.fit


def RFPipeline_PCA(df1, df2, n_iter, cv):
    """
    Creates pipeline that perform Random Forest classification on the data with Principal Component Analysis. The
    input data is split into training and test sets, then a Randomized Search (with cross-validation) is performed to
    find the best hyperparameters for the model.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Dataframe containing the features.
    df2 : pandas.DataFrame
        Dataframe containing the labels.
    n_iter : int
        Number of parameter settings that are sampled.
    cv : int
        Number of cross-validation folds to use.

    Returns
    -------
    pipeline_PCA : sklearn.pipeline.Pipeline
        A fitted pipeline (includes PCA, hyperparameter optimization using RandomizedSearchCV and a Random Forest
        Classifier model).

    See Also
    --------
    PCA : https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    RandomizedSearchCV : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    """

    X = df1.values
    y = df2.values
    region = list(df1.columns.values)

    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.1, random_state=6)

    pipeline_PCA = Pipeline(steps=[("dim_reduction", PCA()),
                                   ("hyper_opt", RandomizedSearchCV(RandomForestClassifier(),
                                                                    param_distributions=param_dist,
                                                                    n_iter=n_iter,
                                                                    cv=cv,
                                                                    random_state=9))
                                   ]
                            )

    pipeline_PCA.fit(X_tr, y_tr)

    y_pred = pipeline_PCA.predict(X_tst)
    y_prob = pipeline_PCA.predict_proba(X_tst)

    # SCORES WITH 95% CONFIDENCE INTERVAL
    scores = performance_scores(y_tst, y_pred, y_prob)

    print("Components shape is:", np.shape(pipeline_PCA["dim_reduction"].components_)[0])

    return pipeline_PCA.fit


def SVM_simple(df1, df2, ker: str):
    """
    Performs SVM classification on the data. The input data is split into training and test sets, then a Grid Search
    (with cross-validation) is performed to find the best hyperparameters for the model. Feature reduction is not
    implemented in this function.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Dataframe containing the features.
    df2 : pandas.DataFrame
        Dataframe containing the labels.
    ker : str
        Kernel type.

    Returns
    -------
    grid : sklearn.model_selection.GridSearchCV
        A fitted grid search object with the best parameters for the SVM model.

    See Also
    --------
    GridSearchCV : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    """
    
    if ker == 'linear':
        param_grid = {'C': stats.expon.rvs(size=100),
                      'gamma': stats.expon(scale=.1).rvs(size=100),
                      'kernel': [ker],
                      'class_weight': ['balanced', None]}

    else:
        param_grid = {'C': np.logspace(-2, 10, 13),
                      'gamma': np.logspace(-9, 3, 13),
                      'kernel': [ker],
                      'class_weight': ['balanced', None]}

    X = df1.values
    y = df2.values
    
    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.1, random_state=6)

    clf = svm.SVC(kernel=ker)

    grid = GridSearchCV(clf, param_grid, refit=True, n_jobs=-1)

    # fitting the model for grid search 
    grid.fit(X_tr, y_tr) 
     
    y_pred = grid.predict(X_tst)
    y_prob = grid.decision_function(X_tst)

    # SCORES WITH 95% CONFIDENCE INTERVAL
    scores = performance_scores(y_tst, y_pred, y_prob)
        
    return grid.fit


def SVM_feature_reduction(df1, df2):
    """
    Performs SVM classification on the data. The input data is split into training and test sets, then a Grid Search
    (with cross-validation) is performed to find the best hyperparameters for the model. Feature reduction is
    implemented in this function.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Dataframe containing the features.
    df2 : pandas.DataFrame
        Dataframe containing the labels.
    Returns
    -------
    grid : sklearn.model_selection.GridSearchCV
        A fitted grid search object with the best parameters for the SVM model using the selected features.

    See Also
    --------
    RFECV: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
    GridSearchCV : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    """

    X = df1.values
    y = df2.values
    
    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.1, random_state=6)

    C_range = stats.expon.rvs(size=8)
    g = stats.expon(scale=.1)
    gamma_range = g.rvs(size=8)

    # defining parameter range 
    param_grid = {'estimator__C': C_range,
                  'estimator__gamma': gamma_range, 
                  'estimator__kernel': ['linear'], 
                  'estimator__class_weight': ['balanced', None]}
    
    clf = svm.SVC(kernel="linear")
    
    rfecv = RFECV(estimator=clf,
                  step=1,
                  cv=5,
                  scoring="accuracy",
                  min_features_to_select=len(y),
                  n_jobs=-1)
    
    grid = GridSearchCV(rfecv, param_grid, refit=True, n_jobs=-1)
    
    # fitting the model
    grid.fit(X_tr, y_tr)
     
    y_pred = grid.predict(X_tst)
    y_prob = grid.decision_function(X_tst)

    # SCORES WITH 95% CONFIDENCE INTERVAL
    scores = performance_scores(y_tst, y_pred, y_prob)

    return grid.fit
