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

C_range=scipy.stats.expon.rvs(size=100)
g=scipy.stats.expon(scale=.1)
gamma_range=g.rvs(size=100)




param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

def RFPipeline_noPCA(a,c,n_iter,cv):
    

#a and c should be pandas dataframe

    
    X=a.values
    y=c.values
    region = list(a.columns.values)

    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.1, random_state=6)

    pipeline_simple = Pipeline(steps=[("hyper_opt", RandomizedSearchCV(RandomForestClassifier(), param_distributions = param_dist, n_iter=n_iter, cv=cv))])
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
        dot_data = export_graphviz(tree,feature_names=region,  filled=True, impurity=False, proportion=True,class_names=["CN","AD"])
        graph = graphviz.Source(dot_data)
        graph.render(view=True)


    return pipeline_simple.fit




def RFPipeline_PCA(a,c,n_iter,cv):


#a and c should be pandas dataframe

    X=a.values
    y=c.values


    region = list(a.columns.values)


    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.1, random_state=6)

    pipeline_PCA = Pipeline(steps=[("dim_reduction", PCA()),("hyper_opt", RandomizedSearchCV(RandomForestClassifier(), param_distributions = param_dist, n_iter=n_iter, cv=cv))])
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
    
    
    print("Components shape is:",np.shape(pipeline_PCA["dim_reduction"].components_)[0])
        
    

    return pipeline_PCA.fit




def SVMPipeline(a,c,ker:str):
    
    param_grid = {'C': C_range,
              'gamma': gamma_range, 
              'kernel': [ker], 
              'class_weight':['balanced', None]}
    
    X=a.values
    y=c.values
    
    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.1, random_state=6)
    
    
    clf = svm.SVC(kernel=ker)

    grid = GridSearchCV(clf, param_grid, refit = True,n_jobs=-1) 

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
    