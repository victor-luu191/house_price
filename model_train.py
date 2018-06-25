import argparse
import os
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

from setting import *

SEED = 1


class Trainer():
    def __init__(self, X, y, test_ratio, stratify=None):

        if not stratify:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_ratio, random_state=SEED)
        else:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_ratio, stratify=stratify, random_state=SEED)

        self.X_train, self.y_train = X_train, y_train
        self.X_valid, self.y_valid = X_valid, y_valid

        lin_models = set_linear_models()
        tree_models = set_tree_models()
        self.models = {**lin_models, **tree_models} # join two dicts
        self.predictions = dict()

    def benchmark(self, metrics_file, pred_file):

        errors = dict()
        for name, predictor in self.models.items():
            errors[name] = self.eval(predictor, name)

        df_error = pd.DataFrame({'model': list(errors.keys()), 'mean_squared_log_error': list(errors.values())})
        df_error.to_csv(metrics_file, index=False)
        print('dumped errors to file {}'.format(metrics_file))

        predict_df = self.retrieve_predictions()
        predict_df.to_csv(pred_file, index=False)
        print('dumped predictions to file {}'.format(pred_file))

        return None

    def eval(self, predictor, name):
        predictor.fit(self.X_train, self.y_train)
        y_pred = predictor.predict(self.X_valid)

        self.predictions[name] = y_pred

        try:
            error = metrics.mean_squared_log_error(self.y_valid, y_pred)
            return error
        except ValueError as e:
            return np.nan

    def retrieve_predictions(self):
        predict_df = self.X_valid
        predict_df['SalePrice'] = self.y_valid
        for predictor in self.models.keys():
            predict_df['price_predict_{}'.format(predictor)] = self.predictions[predictor]

        return predict_df


def set_linear_models():
    # lin_reg = linear_model.LinearRegression()
    ridge_reg = linear_model.Ridge(random_state=SEED)
    lasso_reg = linear_model.Lasso(random_state=SEED)
    lin_predictors = [lasso_reg, ridge_reg]  # lin_reg, ridge_reg
    lin_names = ['Lasso Regression', 'Ridge Regression']  # 'Linear Regression'
    return dict(zip(lin_names, lin_predictors))


def set_tree_models():
    gb_reg = GradientBoostingRegressor(random_state=SEED)
    rf_reg = RandomForestRegressor(random_state=SEED)
    xgb_reg = XGBRegressor()

    tree_predictors = [gb_reg, rf_reg, xgb_reg]
    tree_names = ['Boosted Regression Tree', 'Random Forest',
                  'XGBoost']

    return dict(zip(tree_names, tree_predictors))


def choose_features(df, cat_feat):
    '''
    Choose features to be used as predictors, include the given categorical feature (if any)
    :param df:
    :param cat_feat: given categorical feature
    :return:
    '''
    numerical_feats = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd'
                       ]    # 'age_in_year', 'years_from_remodel'
    if cat_feat != '':
        print('include categorical feature {}'.format(cat_feat))
        area_feats = [ff for ff in df.columns if '{}_'.format(cat_feat) in ff]
        feats = numerical_feats + area_feats
    else:
        feats = numerical_feats

    return feats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        help='name of input file, e.g. data_all.csv'
                        )
    parser.add_argument('--metrics_file',
                        help='name of file for saving metrics, e.g. metrics.csv'
                        )
    parser.add_argument('--pred_file',
                        help='name of file for saving predictions, e.g. predictions.csv')

    return parser.parse_args()

if __name__ == '__main__':
    # args = parse_args()
    # input_file = os.path.join(DAT_DIR, vars(args)['input_file'])
    # metrics_file = os.path.join(RES_DIR, vars(args)['metrics_file'])
    # pred_file = os.path.join(RES_DIR, vars(args)['pred_file'])

    input_file = os.path.join(DAT_DIR, 'data_all.csv')
    data_all = pd.read_csv(input_file)
    print('loaded all data')

    n_train, n_test = 1460, 1459  # TODO replace these hard code values
    train = data_all.iloc[:n_train]

    cat_feat = '' # 'Neighborhood'
    feats = choose_features(data_all, cat_feat)
    print('features used for training models: {}'.format(feats))

    cols = feats + ['SalePrice']
    valid_train = train[cols].dropna()

    trainer = Trainer(X=valid_train[feats], y=valid_train['SalePrice'], test_ratio=0.1)
    metrics_file = os.path.join(RES_DIR, 'metrics_origin.csv')  # 'metrics_{}.csv'.format(cat_feat)
    pred_file = os.path.join(RES_DIR, 'predictions_origin.csv') # 'predictions_{}.csv'.format(cat_feat)

    trainer.benchmark(metrics_file, pred_file)

