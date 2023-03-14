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
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_curve, auc



param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

def RFPipeline_noPCA(X_tr,y_tr,X_tst,y_tst,n_iter,cv):
    
    
    pipeline_simple = Pipeline(steps=[("hyper_opt", RandomizedSearchCV(RandomForestClassifier(), param_distributions = param_dist, n_iter=n_iter, cv=cv))])
    
    
    return pipeline_simple




def RFPipeline_PCA(X_tr,y_tr,X_tst,y_tst,n_iter,cv):
    

    pipeline_PCA = Pipeline(steps=[("dim_reduction", PCA()),("hyper_opt", RandomizedSearchCV(RandomForestClassifier(), param_distributions = param_dist, n_iter=n_iter, cv=cv))])
    
    
    return pipeline_PCA





def plot_cv_roc(X, y, classifier, n_splits=5):
    """
    Train the classifier on X data with y labels and implement the
    k-fold-CV with k=n_splits. Plot the ROC curves for each k fold
    and their average and display the corresponding AUC values
    and the standard deviation over the k folders.
    Parameters
    ----------
    X : numpy array
        Numpy array of read image.
    y : numpy array
        Numpy array of labels.
    classifier : keras model
        Name of wavelet families.
    n_splits : int
        Number of folders for K-cross validation.
    Returns
    -------
    fig : Figure
        Plot of Receiver operating characteristic (ROC) curve.
    Examples
    --------
    >>> classifier = RandomForestClassifier()
    >>> N_FOLDS = 5
    >>> plot_cv_roc(X, y, classifier, n_splits=N_FOLDS):
    """

    try:
        y = y.to_numpy()
        X = X.to_numpy()
    except AttributeError:
        pass

    cv = StratifiedKFold(n_splits)

    tprs = [] #True positive rate
    aucs = [] #Area under the ROC Curve
    interp_fpr = np.linspace(0, 1, 100)
    plt.figure()
    i = 0
    for train, test in cv.split(X, y):

        model = classifier
        prediction = model.fit(X[train], y[train])

        y_test_pred = model.predict(X[test])

        # Compute ROC curve and area under the curve
        fpr, tpr, _ = roc_curve(y[test], y_test_pred)
        interp_tpr = interp(interp_fpr, fpr, tpr)
        tprs.append(interp_tpr)

        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i} (AUC = {roc_auc:.2f})')
        i += 1

    plt.legend()
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.show()

    plt.figure()
    plt.plot([0, 1], [0, 1],
            linestyle='--',
            lw=2,
            color='r',
            label='Chance',
            alpha=.8
    )

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(interp_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(interp_fpr, mean_tpr,
                        color='b',
                        label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
                        lw=2,
                        alpha=.8
                        )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(interp_fpr,
                     tprs_lower,
                     tprs_upper,
                     color='grey',
                     alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.title('Cross-Validation ROC of RandomForestClassifier',fontsize=18)
    plt.legend(loc="lower right", prop={'size': 15})
    plt.show()