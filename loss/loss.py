import numpy as np
from engine.engine import Value

def categorical_cross_entropy(y_true, y_pred:Value):
    if y_true.shape != y_pred.data.shape:
        raise ValueError(f"shape mismatched,{y_true.shape} != {y_pred.data.shape}")

    epsilon = 1e-7
    y_pred = y_pred + epsilon
    y_true = Value(y_true)
    log_pred = y_pred.log()
    elementwise = -y_true * log_pred
    loss = elementwise.mean()
    return loss

def binary_cross_entropy(y_true, y_pred: Value):
    epsilon = 1e-7
    y_true = Value(y_true)
    
    loss = -y_true * (y_pred + epsilon).log() - (1 - y_true) * (1 - y_pred + epsilon).log()
    
    return loss.mean()

def mean_squared_error(y_true, y_pred:Value):
    y_true = Value(y_true)
    error = (y_true - y_pred) **2
    loss = error.mean()
    return loss