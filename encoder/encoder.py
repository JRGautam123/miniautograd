import numpy as np

def one_hot_encoder(y, num_classes=None):
    """
    One hot encoding for target variable.
    """
    classes = sorted(set(y.tolist()))
    classes_idx = {c:i for i, c in enumerate(classes)}
    
    y_encoded = np.array([classes_idx[label] for label in y.tolist()])

    if not num_classes:
        num_classes = len(classes)
    N = len(y_encoded)
    one_hot = np.zeros((N,num_classes))
    one_hot[np.arange(N), y_encoded] = 1.0
    return one_hot

def label_encoder(y):
    """label encoder for target variable"""
    classes = sorted(set(y.tolist()))
    classes_idx = {c:i for i, c in enumerate(classes)}
    y_encoded = np.array([classes_idx[label] for label in y.tolist()])
    return y_encoded.reshape(-1,1)