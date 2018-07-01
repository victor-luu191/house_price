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
    res = pd.concat([df.drop(feat, axis=1), encoded], axis='columns')
    return res


def to_full(abbv_feat, df, full_form):
    # convert an abbreviated feature in data into its full form via given full forms
    full_feat = 'full_' + abbv_feat
    df[abbv_feat] = df[abbv_feat].fillna("NA")
    df[full_feat] = df[abbv_feat].apply(lambda x: full_form[x])
    return df


def load_full_form(fname):
    tmp = pd.read_csv(os.path.join(DAT_DIR, fname), keep_default_na=False)
    full_form = dict(zip(tmp['abbr'], tmp['full']))

    # debug
    # print('full forms: {}'.format(full_form))
    return full_form


if __name__ == '__main__':
    train = pd.read_csv(os.path.join(DAT_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DAT_DIR, 'test.csv'))
    # this is applicable only to well-indexed data sets
    print('last row in train set: {}'.format(max(train.Id)))
    print('first row in test set: {}'.format(min(test.Id)))

    ## Preprocesses
    response = 'SalePrice'
    data_all = join(train, test, response)

    print('Add derived features...')
    data_all['age_in_year'] = data_all['YrSold'] - data_all['YearBuilt']
    data_all['years_from_remodel'] = data_all['YrSold'] - data_all['YearRemodAdd']

    data_all = onehot_encode('Neighborhood', data_all)
    zone_full = load_full_form('zones.csv')
    data_all = to_full('MSZoning', data_all, zone_full)
    data_all = onehot_encode('full_MSZoning', data_all)

    ## End of preprocesses ==================
    print('Shape of data_all after all preprocessing: {}'.format(data_all.shape))

    fname = os.path.join(DAT_DIR, 'data_all.csv')
    data_all.to_csv(fname, index=False)
    print('Save processed data to file {}'.format(fname))