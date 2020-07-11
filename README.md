# House Price Prediction (WIP)

This repo contains source code of building models for predicting house price in [this Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). It illustrates nicely that careful feature selection can lead to very good prediction without having to resort to complicated models. Such a nice selection can easily help me climb more than 500 ranks on leaderboard.

If you find this repo useful, __kindly give it a star__ :). Happy learning.

## What's in this for you?

You will find:
+ an insightful EDA notebook

+ a `predictive model` notebook with decent examples of iterative model building via adding features incrementally

+ a decent notebook on feature engineering with effective ways to transform features and target to achieve better prediction performance. 

+ nice way/tool to measure relationship between categorical variable and continuous target

+ smart encoding schemes for categorical variables

+ general source codes which can be reused for other prediction projects.

+ clear observation on the impact of feature engineering tasks on prediction. It is especially good for beginners to DS and ML.

## Major classes
+ `DataPrep`: for all data preprocessing such as filling NAs, onehot encoding categorical features, encoding quantitative features which are disguised as text
+ `Trainer`: for training and model selection via cross-validation
+ `Predictor`: for making predictions
+ `PostAnalyzer`: for post-train analysis
