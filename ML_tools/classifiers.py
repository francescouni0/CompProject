from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from IPython.display import display 
import graphviz
import numpy as np
from sklearn import svm, metrics
from sklearn.feature_selection import RFECV
import scipy
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(os.getcwd()).parent))



param_dist = {'n_estimators': randint(50, 500),
              'max_depth': randint(1, 20)}


def RFPipeline_noPCA(df1, df2, n_iter, cv):
    """Scikit pipeline that performs a Random Forest classification on the data without 
    principal compoenent analysis. The input data is split into training and test sets,
    then a Randomized search is performed to find the best hyperparameters for the model.
    
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    
    
    Args:
        df1 (Pandas datafrae): Dataframe that contains the features
        df2 (Pandas Dataframe): Dataframe that contains the labels
        n_iter (int): Number of iterations for the Randomized search
        cv (int): Determines the cross-validation splitting strategy.

    Returns:
        object: fitted pipeline
    """

    X = df1.values
    y = df2.values
    region = list(df1.columns.values)

    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.1, random_state=6)

    pipeline_simple = Pipeline(steps=[("hyper_opt", RandomizedSearchCV(RandomForestClassifier(),
                                                                       param_distributions=param_dist,
                                                                       n_iter=n_iter,
                                                                       cv=cv,
                                                                       random_state=9))
                                      ]
                               )

    pipeline_simple.fit(X_tr, y_tr)

    y_pred = pipeline_simple.predict(X_tst)

    accuracy = accuracy_score(y_tst, y_pred)
    precision = precision_score(y_tst, y_pred)
    recall = recall_score(y_tst, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    ax = plt.gca()
    rnds_disp = RocCurveDisplay.from_estimator(pipeline_simple, X_tst, y_tst, ax=ax, alpha=0.8)
    plt.show()

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
    """Scikit pipeline that performs a Random Forest classification on the data with 
    principal compoenent analysis. The input data is split into training and test sets,
    then a Randomized search is performed to find the best hyperparameters for the model.
    
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    
    Args:
        df1 (Pandas datafrae): Dataframe that contains the features
        df2 (Pandas Dataframe): Dataframe that contains the labels
        n_iter (int): Number of iterations for the Randomized search
        cv (int): Determines the cross-validation splitting strategy.

    Returns:
        object: Fitted pipeline
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

    accuracy = accuracy_score(y_tst, y_pred)
    precision = precision_score(y_tst, y_pred)
    recall = recall_score(y_tst, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    ax = plt.gca()
    rnds_disp = RocCurveDisplay.from_estimator(pipeline_PCA, X_tst, y_tst, ax=ax, alpha=0.8)
    plt.show()

    print("Components shape is:", np.shape(pipeline_PCA["dim_reduction"].components_)[0])

    return pipeline_PCA.fit


def SVMPipeline(df1, df2, ker: str):
    """Pipeline that performs a SVM classification on the data. The input data is split into
    training and test sets, than the best hyperparameters are found using a grid search. There is 
    not a feature reduction step in this pipeline.

    Args:
        df1 (Pandas Dataframe): Dataframe that contains the features
        df2 (Pandas Dataframe): Dataframe that contains the labels
        ker (str): kernel type

    Returns:
        object: Fitted pipeline
    """
    
    if ker == 'linear':
        param_grid = {'C': scipy.stats.expon.rvs(size=100),
                      'gamma': scipy.stats.expon(scale=.1).rvs(size=100),
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
     
    # print best parameter after tuning 
    print(grid.best_params_) 
    
    grid_predictions = grid.predict(X_tst) 
    print(grid_predictions)
    print(y_tst)
       
    # print classification report 
    print(metrics.classification_report(y_tst, grid_predictions)) 
        
    return grid.fit


def SVMPipeline_feature_red(df1, df2):
    """Pipeline that performs a SVM classification on the data. The input data is split into
    training and test sets, than the best hyperparameters are found using a grid search. There is a
    feature reduction step in this pipeline.
    Args:
        df1 (Pandas Dataframe): Dataframe that contains the features
        df2 (Pandas Dataframe): Dataframe that contains the labels

    Returns:
        object: Fitted pipeline
    """

    X = df1.values
    y = df2.values
    
    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.1, random_state=6)

    C_range = scipy.stats.expon.rvs(size=10)
    g = scipy.stats.expon(scale=.1)
    gamma_range = g.rvs(size=10)

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
     
    # print best parameter after tuning 
    predictions = grid.predict(X_tst) 
    print(predictions)
    print(y_tst)
       
    # print classification report 
    print(metrics.classification_report(y_tst, predictions))

    return grid.fit
    
        
        