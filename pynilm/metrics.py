from sklearn.metrics import *

# Custom metric (f1-macro)
def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')