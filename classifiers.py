from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz




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
        tree = RandomizedSearchCV.best_estimator_[i]
        dot_data = export_graphviz(tree,feature_names=region[1:],  filled=True, impurity=False, proportion=True,class_names=["CN","AD"])
        graph = graphviz.Source(dot_data)
        display(graph)


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
    
    for i in range(3):
        tree = RandomizedSearchCV.best_estimator_[i]
        dot_data = export_graphviz(tree,feature_names=region[1:],  filled=True, impurity=False, proportion=True,class_names=["CN","AD"])
        graph = graphviz.Source(dot_data)
        display(graph)

    return pipeline_PCA.fit