# Dealing with categorical variables (WIP)

## Recap
Though they appear in data as text values, categorical variables actually have two different types: _ordinal_ vs. _nominal_. The former inherently has some order, e.g. "high" > "medium" > "low", while the latter are simply names of different categories and have no order, e.g. names of colors. Thus, we need to handle the two types different.

This note will talk about:

+ metrics for quantifying relationship between a categorical variable and a numeric variable; between two categorical variables

+ different encoding schemes for categorical variables

This note will focus on practical aspects, so that readers can apply right away, theoritical discussion are not in scope.

## Ordinal variables

+ encoders: `LabelEncoder` in `sklearn`.

## Nominal variables

Metrics for __association__ between _nominal_ variables _x_ and _y_.
+ Cramer V: symmetry metrics
+ Theil's U: aka uncertainty coefficient, is based on conditional entropy between _x_ and _y_ ---  or in human language, given the value of x, how many possible states does y have, and how often do they occur. 

### Implementation
`dython` package

## Encoding schemes

+ Count encoding
+ Target encoding
+ Catboost encoding