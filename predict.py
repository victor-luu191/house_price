from sklearn.externals import joblib

from setting import *


class Predictor():
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict(self, X):
        feats = self.features
        return self.model.predict(X[feats])


def load_predictor(fname):
    predictor = joblib.load(fname)
    return predictor


def load_test_data():
    print('Load test data')
    input_file = os.path.join(DAT_DIR, 'data_all.csv')
    data_all = pd.read_csv(input_file)
    X_test = data_all[data_all['SalePrice'].isnull()]
    return X_test


if __name__ == '__main__':
    fname = os.path.join(MODEL_DIR, 'boosted_regression_tree.pkl') # random_forest, xgboost,
    predictor = load_predictor(fname)

    X_test = load_test_data()

    X_test['SalePrice'] = predictor.predict(X_test)
    X_test.drop_duplicates(inplace=True)
    pred_file = os.path.join(RES_DIR, 'predictions.csv')
    X_test.to_csv(pred_file, index=False)

    print('make submission file')
    cols = ['Id', 'SalePrice']
    submit = X_test[cols].drop_duplicates() # todo check why duplicates occur
    submit.to_csv(os.path.join(RES_DIR, 'submit.csv'), index=False)