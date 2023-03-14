from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import RocCurveDisplay
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt




param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

def RFPipeline_noPCA(X_tr,y_tr,X_tst,y_tst,n_iter,cv):




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


    return pipeline_simple.fit




def RFPipeline_PCA(X_tr,y_tr,X_tst,y_tst,n_iter,cv):




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

    return pipeline_PCA.fit