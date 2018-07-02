import pandas as pd
import numpy as np
import os

from predict import load_predictor
from setting import MODEL_DIR


class ErrorAnalyzer():
    def get_feature_importance(self, predictor):
        feats = predictor.features
        feature_importances = predictor.model.feature_importances_
        res = pd.DataFrame({'feature': feats, 'importance': feature_importances})
        return res.sort_values('importance', ascending=False)

if __name__ == '__main__':
    fname = os.path.join(MODEL_DIR, 'boosted_regression_tree.pkl')  # random_forest, xgboost
    predictor = load_predictor(fname)



    # tmp = dir(predictor.model)
    # print([attr for attr in tmp if 'feature' in attr])