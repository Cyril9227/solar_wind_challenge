from sklearn.base import BaseEstimator
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np





class Classifier(BaseEstimator):
    def __init__(self):
        None
    
    def fit(self, X, y):
        self.model = LGBMClassifier()
        self.model.fit(X, y)

    def predict_proba(self, X):
        y_pred = self.model.predict_proba(X)
        s_y_pred = pd.Series(y_pred[:, 1])
        s_y_pred_smoothed = s_y_pred.rolling(80, min_periods=0, center=True).quantile(0.6)
        return np.array(list(zip(1-s_y_pred_smoothed[:],s_y_pred_smoothed[:])))
    
    
