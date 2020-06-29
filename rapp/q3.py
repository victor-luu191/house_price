# Please write a function which has the following arguments:
# a.  Input dataframe (df)
# b. a vector of column names that list the quantitative variables to be analyzed.
# c. a vector called funs, that describe the type of summaries to perform
# (see below by the list of summaries to perform)
# d. a group by variable that would allow the end user to condition out these statistics by
# levels of a categorical variable.

# “funs” arguments:
# “sum”, “mean”, “max”, “min”

# The function should return a data frame with each of the analyzed column names
# summarized by each measures with 1 row per relevant level of the group by variable.
# Columns should be named so that their meaning is intuitive to
# the reader of the function of the output.

import pandas as pd


def flatten_columns(df):
    df.columns = ['_'.join(col[::-1]).strip() for col in df.columns.values]
    return df


def summarize(df: pd.DataFrame, col_names: list, group_by_var='a categorical variable',
              funs=['sum', 'mean']) -> pd.DataFrame:
    res = df[col_names + [group_by_var]].groupby(group_by_var).aggregate(funs)
    return flatten_columns(res).reset_index()


if __name__ == '__main__':
    df = pd.read_csv('hsbdemo.csv')
    summary_df = summarize(df)
