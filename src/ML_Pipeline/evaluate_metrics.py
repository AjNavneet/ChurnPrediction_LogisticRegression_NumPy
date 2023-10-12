import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

# Function to create a confusion matrix
def confusion_matrix(y_test, y_pred):
    """
    Create and display a confusion matrix.

    Parameters:
    y_test (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    None
    """
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    print('True Negative (TN): %0.2f' % tn)
    print('True Positive (TP): %0.2f' % tp)
    print('False Positive (FP): %0.2f' % fp)
    print('False Negative (FN): %0.2f' % fn)

# Function for creating an ROC curve
def roc_curve(logreg, X_test, y_test):
    """
    Create and display an ROC curve.

    Parameters:
    logreg: The logistic regression model.
    X_test (array-like): Test data.
    y_test (array-like): True labels.

    Returns:
    None
    """
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict(X_test))

    # Setting the graph area
    plt.figure()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Plotting the worst line possible
    plt.plot([0, 1], [0, 1], 'b--')

    # Plotting the logistic regression we have built
    plt.plot(fpr, tpr, color='darkorange', label='Logistic Regression (AUC = %0.2f)' % logit_roc_auc)

    # Adding labels and legend
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    # plt.show()  # Uncomment this line if you want to display the ROC curve
