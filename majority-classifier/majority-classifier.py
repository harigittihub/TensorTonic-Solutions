import numpy as np

def majority_classifier(y_train, X_test):
    # Find the most common label
    majority_label = np.bincount(y_train).argmax()

    # Predict the majority label for all test samples
    return np.full(len(X_test), majority_label)