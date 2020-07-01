# House Price Prediction (WIP)

This repo contains source code of creating models for predicting house price in [this Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). 

It contains:
+ insightful EDA
+ a decent example of iterative model building via adding features incrementally
+ nice way to measure relationship between categorical variable and ...

## What's in this for you?
+ The source codes are general and thus can be reused for other prediction projects.
+ Features are added incrementally so that their impact on prediction can be observed crystally clear. It is especially good for beginners to DS and ML.
+ Post-train analysis are provided so that you can dive deeper

## Major classes
+ `DataPrep`: for all data preprocessing such as filling NAs, onehot encoding categorical features, encoding quantitative features which are disguised as text
+ `Trainer`: for training and model selection via cross-validation
+ `Predictor`: for making predictions
+ `PostAnalyzer`: for post-train analysis
