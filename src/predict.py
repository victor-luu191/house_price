from sklearn.externals import joblib

from src.setting import *
import os


class Predictor():
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict_(self, test):
        feats = self.features
        return self.model.predict(test[feats])

    def refit(self, train):
        X_train, y_train = train[self.features], train['SalePrice']
        self.model.fit(X_train, y_train)
        return self


def load_predictor(fname):
    predictor = joblib.load(fname)
    return predictor


def load_data(fname):
    print('Load data')
    input_file = os.path.join(DAT_DIR, fname)
    data_all = pd.read_csv(input_file)
    return data_all


if __name__ == '__main__':
    fname = os.path.join(MODEL_DIR, 'boosted_regression_tree.pkl')  # random_forest, xgboost,
    predictor = load_predictor(fname)

    data = load_data('data_all.csv')
    train = data[data['SalePrice'].notnull()]
    test = data[data['SalePrice'].isnull()]

    predictor = predictor.refit(train)

    test['SalePrice'] = predictor.predict_(test)
    test.drop_duplicates(inplace=True)
    pred_file = os.path.join(RES_DIR, 'predictions.csv')
    test.to_csv(pred_file, index=False)

    print('make submission file')
    cols = ['Id', 'SalePrice']
    submit = test[cols]
    print('# rows in submit: {}'.format(submit.shape[0]))
    submit.to_csv(os.path.join(RES_DIR, 'submit.csv'), index=False)
