# Generic ML models
from sklearn.neural_network import MLPClassifier 
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier

SAMPLE_RATE = 2 # Frequencia do sinal (LF)

APPLIANCES = ['washer dryer', 'microwave', 'dish washer', 'fridge']

# Instantiate pipeline resources
# # DTLFE params
RP_PARAMS = {
    "dimension": 1,
    "time_delay": 1,
    "threshold": None,
    "percentage": 10
}

IMG_SIZE = 32

INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1)

# # Estimator params
SEED = 33

ESTIMATORS = {
    "MLP": MLPClassifier(alpha=1e-3, hidden_layer_sizes=(10,), random_state=SEED),
    "SVM": SVC(kernel='rbf', random_state=SEED),
    "XGBOOST": XGBClassifier(random_state=SEED, n_jobs=4)
}

EVALUATION_METRICS = {
    'accuracy': 'accuracy_score', 
    'f1': 'f1_macro',
    'precision': 'precision_score', 
    'recall':  'recall_score'
}

# Window size (sliding window)
# Eg.: 1min power consumption, at 2s sample rate => length = 30
WINDOW_SIZE = int((1 * 60) / SAMPLE_RATE) # (minutes * 60) / sample rate

WINDOW_STRIDE = WINDOW_SIZE  # no overlapping