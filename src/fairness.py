import pandas as pd
from fairlearn.metrics import (MetricFrame, selection_rate,
                                false_positive_rate, false_negative_rate,
                                demographic_parity_difference,
                                equalized_odds_difference)


def audit(y_test, y_pred, sensitive_features):
    metrics = {
        'selection_rate': selection_rate,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate
    }
    mf = MetricFrame(metrics=metrics, y_true=y_test,
                     y_pred=y_pred, sensitive_features=sensitive_features)
    dp = demographic_parity_difference(y_test, y_pred,
                                        sensitive_features=sensitive_features)
    eo = equalized_odds_difference(y_test, y_pred,
                                    sensitive_features=sensitive_features)
    return mf, dp, eo


def summary(y_test, baseline_pred, fair_pred, sensitive_features):
    _, dp_base, eo_base = audit(y_test, baseline_pred, sensitive_features)
    _, dp_fair, eo_fair = audit(y_test, fair_pred, sensitive_features)

    return pd.DataFrame({
        'Metric': ['Demographic Parity Diff', 'Equalized Odds Diff'],
        'Baseline': [round(dp_base, 3), round(eo_base, 3)],
        'Fair Model': [round(dp_fair, 3), round(eo_fair, 3)]
    })