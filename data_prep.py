import pandas as pd
import os
import numpy as np
from setting import DAT_DIR


def join(train, test, response):
    test[response] = np.nan
    return pd.concat([train, test])


def onehot_encode(feat, df):
    print('Onehot encode feature {}'.format(feat))
    encoded = pd.get_dummies(df[feat], prefix=feat)
    res = df.drop(feat, axis=1).join(encoded)
    return res

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(DAT_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DAT_DIR, 'test.csv'))

    # this is applicable only to well-indexed data sets
    print('last row in train set: {}'.format(max(df.Id)))
    print('first row in test set: {}'.format(min(test.Id)))

    response = 'SalePrice'
    data_all = join(df, test, response)

    print('Add derived features...')
    data_all['age_in_year'] = data_all['YrSold'] - data_all['YearBuilt']
    data_all['years_from_remodel'] = data_all['YrSold'] - data_all['YearRemodAdd']

    data_all = onehot_encode('Neighborhood', data_all)

    print('Shape of data_all after preprocessing: {}'.format(data_all.shape))

    fname = os.path.join(DAT_DIR, 'data_all.csv')
    data_all.to_csv(fname, index=False)
    print('Save processed data to file {}'.format(fname))