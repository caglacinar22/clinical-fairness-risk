import pandas as pd
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds


def train_baseline(X_train, y_train):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_fair(X_train, y_train, sensitive_train):
    estimator = LogisticRegression(random_state=42, max_iter=1000)
    constraint = EqualizedOdds()
    fair_model = ExponentiatedGradient(estimator, constraint)
    fair_model.fit(X_train, y_train, sensitive_features=sensitive_train)
    return fair_model