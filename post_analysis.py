import pandas as pd

from predict import load_predictor
from setting import MODEL_DIR


class PostAnalyzer():
    '''
    Object for post-train analysis
    '''
    def __init__(self):
        pass

    def get_feature_importance(self, predictor):
        feats = predictor.features
        feature_importances = predictor.model.feature_importances_
        res = pd.DataFrame({'feature': feats, 'importance': feature_importances})
        return res.sort_values('importance', ascending=False)

    def top_k_features(self, k, predictor):
        feat_importance = self.get_feature_importance(predictor)
        return feat_importance.head(k)

if __name__ == '__main__':
    brt = load_predictor(MODEL_DIR + '/boosted_regression_tree.pkl')
    rf = load_predictor(MODEL_DIR + '/random_forest.pkl')
    xgb = load_predictor(MODEL_DIR + '/xgboost.pkl')

    pa = PostAnalyzer()
    brt_top3 = pa.top_k_features(k=3, predictor=brt)
    print('Top 3 features of models:')
    print('\t boosted_regression_tree: {}'.format(brt_top3))

    rf_top3 = pa.top_k_features(k=3, predictor=rf)
    print('\t random forest: {}'.format(rf_top3))

    xgb_top3 = pa.top_k_features(k=3, predictor=xgb)
    print('\t XG Boost: {}'.format(xgb_top3))

    # tmp = dir(brt.model)
    # print([attr for attr in tmp if 'feature' in attr])