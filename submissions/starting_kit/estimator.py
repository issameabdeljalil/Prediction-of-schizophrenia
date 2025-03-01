#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 02:11:17 2022

@author: edouard.duchesnay@cea.fr
"""

# import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso

remaining_features = [12, 20, 29, 51, 62, 63, 68, 70, 71, 83, 84, 92, 95, 104,
                      119, 130, 137, 9, 16, 27, 37, 45, 46, 49, 50, 54, 55, 57,
                      58, 60, 64, 69, 73, 75, 80, 90, 91, 94, 96, 98, 101, 107,
                      112, 113, 125, 126, 127, 128, 129, 131, 133, 135, 136, 139, 145, 154,
                      158, 189, 195, 197, 199, 200, 205, 210, 211, 212, 215, 222, 228, 231,
                      233, 236, 239, 240, 241, 242, 249, 257, 266, 271, 273, 281]
 
class ROIsFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self
 
    def transform(self, X):
        return X[:, remaining_features]
 
def get_estimator():
    estimator = make_pipeline(
        ROIsFeatureExtractor(),
        StandardScaler(),
        StackingClassifier(
            estimators=[
                ('svc', SVC(probability=True,random_state=1)),
                ('gb', GradientBoostingClassifier(random_state=1)),
                ('mlp', MLPClassifier(random_state=1, hidden_layer_sizes=(200, 150, 100, 50, 25, ))) 
            ],
        )
    )
    return estimator