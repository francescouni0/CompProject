import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc


def performance_scores(y_test, y_predicted, y_probability, confidence_int=0.683):
    """
    Add documentation
    """
    if y_probability.ndim == 2:
        y_prob = y_probability[:, 1]
    else:
        y_prob = y_probability

    z_score = scipy.stats.norm.ppf((1 + confidence_int) / 2.0)

    accuracy = accuracy_score(y_test, y_predicted)
    accuracy_err = z_score * np.sqrt((accuracy * (1 - accuracy)) / y_test.shape[0])

    precision = precision_score(y_test, y_predicted)
    precision_err = z_score * np.sqrt((precision * (1 - precision)) / y_test.shape[0])

    recall = recall_score(y_test, y_predicted)
    recall_err = z_score * np.sqrt((recall * (1 - recall)) / y_test.shape[0])

    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    n1 = sum(y_test == 1)
    n2 = sum(y_test == 0)
    q1 = roc_auc / (2 - roc_auc)
    q2 = 2 * roc_auc ** 2 / (1 + roc_auc)
    auc_err = np.sqrt((roc_auc * (1 - roc_auc) + (n1 - 1) * (q1 - roc_auc ** 2) + (n2 - 1) * (q2 - roc_auc ** 2)) / (n1 * n2))

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC')
    plt.legend(loc="lower right")
    plt.show()

    print("Accuracy:", round(accuracy, 2), "+/-", round(accuracy_err, 2))
    print("Precision:", round(precision, 2), "+/-", round(precision_err, 2))
    print("Recall:", round(recall, 2), "+/-", round(recall_err, 2))
    print("AUC:", round(roc_auc, 2), "+/-", round(auc_err, 2))

    scores = {"Accuracy": accuracy, "Accuracy_error": accuracy_err,
              "Precision": precision, "Precision_error": precision_err,
              "Recall": recall, "Recall_error": recall_err,
              "AUC": roc_auc, "AUC_error": auc_err}

    return scores
