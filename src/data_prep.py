import pandas as pd
import os
import numpy as np
from sklearn.externals import joblib

from src.setting import DAT_DIR


class DataPrep():
    '''
    Data preprocessing object
    '''

    def __init__(self, cat_feats=None, quant_feats=None, scorings=None):
        '''
        Initialize an object for data preprocessing to
        + onehot encode specified categorical features
        + convert quantitative features disguised as text to scores

        :param cat_feats: specified categorical features
        :param quant_feats: specified quantitative features
        :param scorings: list of scoring for each quantitative feature
        '''
        self.cat_feats = cat_feats
        self.quant_feats = quant_feats
        self.scorings = scorings
        pass

    def add_derived_feats(self, data):
        print('Add derived features...')
        data['age_in_year'] = data['YrSold'] - data['YearBuilt']
        data['years_from_remodel'] = data['YrSold'] - data['YearRemodAdd']
        return data

    def encode_cat_feats(self, data):
        cat_feats = self.cat_feats
        print('Onehot encode categorical features {}'.format(cat_feats))

        encoded_df = data.copy()
        # encode 1 cat feature at a time
        for cf in self.cat_feats:
            encoded_df = self.onehot_encode(cf, encoded_df)

        return encoded_df

    def onehot_encode(self, cat_feat, data):
        encoded = pd.get_dummies(data[cat_feat], prefix=cat_feat, dummy_na=True)
        res = pd.concat([data.drop(columns=[cat_feat]), encoded], axis='columns')
        return res

    def quant_to_scores(self, data):
        print('\n Converting quantitative text features to scores...')
        score_dict = dict(zip(self.quant_feats, self.scorings))
        for tf in self.quant_feats: # score_dict.keys()
            data = to_quantitative(text_feat=tf, df=data, scoring=score_dict[tf])

        return data

    def choose_features(self, data):
        '''
        Choose features to be used as predictors from:
        + numerical feats
        + categorical feats
        :param data:
        :return:
        '''
        onehot_feats = self.query_onehot_features(data)
        numerical_feats = self.query_numeric_features()

        features = onehot_feats + numerical_feats
        print('Features used for training models: {}'.format(features))

        self.features = features
        self.numerical_feats = numerical_feats
        self.onehot_features = onehot_feats

    def query_numeric_features(self):
        numerical_feats = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                           'age_in_year', 'years_from_remodel',
                           ]
        area_feats = ['TotalBsmtSF',
                      '1stFlrSF',
                      '2ndFlrSF',
                      'WoodDeckSF',
                      'GrLivArea',
                      'GarageArea',
                      ]
        room_feats = ['TotRmsAbvGrd',
                      ]
        numerical_feats += (area_feats + room_feats)
        # include score features to numerical features
        if self.quant_feats:
            print('Adding score features:')
            score_feats = to_score_feats(self.quant_feats)
            print(score_feats)
            numerical_feats += score_feats
        return numerical_feats

    def query_onehot_features(self, data):
        onehot_features = []
        if self.cat_feats:
            print('Adding onehot-encoded names of categorical features:')
            print(self.cat_feats)
            for cf in self.cat_feats:
                onehot_features += get_onehot_features(cf, data)
        return onehot_features

    def fillna_numeric_feats(self, data, value):
        # fill NA in numerical feats in both train and test sets
        print('Fill NAs in features by {}'.format(value))
        copy = data.copy()
        copy[self.numerical_feats] = data[self.numerical_feats].fillna(value)

        # self.check_na(copy)
        return copy

    def check_na(self, df):
        # check if any NA left
        na_count = [sum(df[ff].isnull()) for ff in self.features]
        print('features still have NA')
        print(pd.DataFrame({'feature': self.features, 'na_count': na_count}).query('na_count > 0'))

    def dump(self):
        # for persistence of features
        fname = os.path.join(DAT_DIR, 'data_prep.pkl')
        joblib.dump(self, fname)


def get_onehot_features(cat_feat, df):
    '''
    Include the given categorical feature
    :param df:
    :param cat_feat: given categorical feature
    :return:
    '''

    onehot_features = [ff for ff in df.columns if '{}_'.format(cat_feat) in ff]
    return onehot_features


def to_score_feats(quant_feats):
    return [qf + '_score' for qf in quant_feats]


def join(train, test, response):
    test[response] = np.nan
    return pd.concat([train, test])


def to_quantitative(text_feat, df, scoring):
    '''
    Given a feature stored in data as text but actually a quantitative feat, convert it to numerical values
    via given encoding
    :param scoring:
    :param text_feat:
    :return:
    '''
    n_na = sum(df[text_feat].isnull())
    print('\t Feature {0} has {1} NAs, they will be filled by 0'.format(text_feat, n_na))

    res = df.copy()
    res[text_feat].fillna("NA", inplace=True)

    # print('\t Column {} has {} NAs, they will be filled by forward filling'.format(text_feat, n_na))
    # res[text_feat].fillna(method='ffill', inplace=True)
    res['{}'.format(text_feat) + '_score'] = res[text_feat].apply(lambda form: scoring[form])
    return res


if __name__ == '__main__':
    train = pd.read_csv(os.path.join(DAT_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DAT_DIR, 'test.csv'))

    ## Preprocesses
    response = 'SalePrice'
    data_all = join(train, test, response)

    cat_feats = ['MSZoning',
                 'Neighborhood',
                 # 'SaleType',
                 'SaleCondition',
                 ]

    six_scale = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
    quant_feats = ['Utilities',
                   'ExterQual',
                   'ExterCond',
                   'HeatingQC',
                   'BsmtQual',
                   'BsmtCond',
                   'KitchenQual',
                   # 'FireplaceQu',
                   'BsmtExposure',
                   'BsmtFinType1',
                   'GarageQual',
                   'GarageCond',
                   ]
    scorings = [{"AllPub": 4, "NoSewr": 3, "NoSeWa": 2, "ELO": 1, "NA": 0},
                six_scale,
                six_scale,
                six_scale,
                six_scale,
                six_scale,
                six_scale,
                # six_scale,
                {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "NA": 0},
                {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "NA": 0},
                six_scale,
                six_scale,
                ]

    dp = DataPrep(cat_feats=cat_feats,
                  quant_feats=quant_feats,
                  scorings=scorings)

    data_all = dp.add_derived_feats(data_all)
    data_all = dp.encode_cat_feats(data_all)
    data_all = dp.quant_to_scores(data_all)

    dp.choose_features(data_all)
    data_all = dp.fillna_numeric_feats(data_all, value=0)
    dp.dump()
    ## End of preprocesses ==================

    print('Shape of data_all after all preprocessing: {}'.format(data_all.shape))


    fname = os.path.join(DAT_DIR, 'data_all.csv')
    data_all.to_csv(fname, index=False)
    print('Save processed data to file {}'.format(fname))
