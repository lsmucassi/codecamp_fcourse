# Free Code Camp Full Data Science Course

### Module 0 - Introduction

#####  What is Data Science
A field that uses scientific methods to extract knowledge and insight from data, the field also has the following specializations:
- Data Engineers: Build software that gathers data from different sources
- Machine Learning scientist/engineer: Develop algorithms. 
- Data Analysts: Answers business questions based on the insight drawn from the data collected.

#####  Data Science Environment setup - basic
- Install & setup Anaconda
    > Anaconda comes with Jupyter - a note made for Data scientist to experiment on 


### Module 1 - Regression

##### simple linear regression 
formula: y = $\alpha$ + $\beta x$
multiple linear regression y = $\alpha$ + $\beta x1$ + $\beta x2$ + $\beta x3$ + ... + $\beta xn$

### Module 2 - Classification

##### simple classification:
Binary classification
classify in more than 2(1:0/True:False) classes
    > example:
        - spam or not
        - fraudulent transaction or not
        - eye colour (blue, green, brown)

Regression VS Classification
 - Regression: response variable is quantitative 
 - Classification: response variable is categorical or qualitative

##### Logistic Regression - probability
- Determine the probability of an observation to be part of a class or not
- Output betwwen 0 and 1 ( 1 means very likely) using sigmoid function

*sigmoid
*logit
*multiple logistic regressions

```
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train.ravel())

y_prob = logistic_reg.predict_proba(X_test)[:,1]
y_pred = np.where(y_prob > 0.5, 1, 0)
```

##### Linear Discriminant Analysis (LDA)
Caveats of logistics regression
    - when classes are well separated, parameters of logistic regression are unstable
    - unstable for small data
    - Not the best to predict more than 2 classes 

What makes LDA unique
    - models the distributions of predictors separately for each class
    - uses Bayes' theorem to estimate the probability

Assumptions of LDA (one predictor)
    - each class is drawn from a Gaussian distribution
    - each class has it's own mean
    - Assume a common variance

Assumptions of LDA (more than one predictor)
    - each class is drawn from a multivariate Gaussian distribution
    - each class has it's own mean vector
    - Assume a common covariance matrix

lda in python
```
lda = LogisticRegression()

lda.fit(X_train, y_train.ravel())

y_prob = logistic_reg.predict_proba(X_test)[:,1]
y_pred = np.where(y_prob > 0.5, 1, 0)
```

### Module 3 - Re-sampling


### Module 4 - Regularization

### Module 5 - Decision Trees

### Module 6 - Support Vectors

### Module 7 - Unsupervised Learning
