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

##### Quadratic Discriminant Analysis (QDA)
Assumptions of QDA (more than one predictor)
    - each class is drawn from a multivariate Gaussian distribution
    - each class has it's own mean vector
    - Assume a common covariance matrix

Advantages of QDA
 - Better for large datasets
 - Lower bias, higher variance
  
qda in python
```
qda = LogisticRegression()

qda.fit(X_train, y_train.ravel())

y_prob_qda = logistic_reg.predict_proba(X_test)[:,1]
y_pred_qda = np.where(y_prob > 0.5, 1, 0)
```
##### How to assess the performance of a model
- Sensitivity: true positive rate. Proportion of actual positives identified.
    > example: proportion of fraudulent transactions that are actually fraudulent 
- Specificity: true negative rate. Proportion of actual negatives identified
    > example: proportion of non-fraudulent transactions that are actually non-fraudulent 

ROC Curve
    - ROC curve (receiver operating characteristic)
    - Take the area under the curve (AUC)
    - The closer to 1, the better

ROC in python
```
def plot_roc(roc_auc):

    plt.figure(figsize=(7, 7))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, c='red', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
```

### Module 3 - Re-sampling

##### re-sampling: 
important for model performance and validation. Done in every data science task
    > example: cross validation

intuition
    - repeatedly draw samples from a training set and refit the model
    - gain more information
    - see how model would perform on new data without collecting new data

cross-validation
    - widely used
    - used to evaluate model's performance and find the best parameters for the model
    uses 3 approaches
        - validation set: randomly split the dataset into training and test set
        - leave-one-out validation: only one data point is used for training and the rest for testing
        - k-fold cross-validation: randomly split the dataset to k-groups/ folds of equal times

k-fold cross validation in python
```
 lin_reg = LinearRegression()

 MSEs = cross_val_score(lin_reg, X, y, scoring='neg_mean_squared_error', cv=5)

 mean_MSE = np.mean(MSEs)

 print(-mean_MSE)
```

### Module 4 - Regularization

##### regularization:
used to prevent overfitting and improve performance
    > example: ridge regression and lasso

intuition
    - models can overfit, meaning that they will not generalize well
    - Poor performance on 'unseen' data

regularization methods
    - Ridge regression
    - Lasso
    Also called: shrinkage methods

Ridge regression
    - lambda is a tuning parameter
    - find the best lambda using cross validation, Use a range of lambdas
    - all predictors are kept
    also called: L2 regularization

ridge regression in python
```
ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(X,y)
```

Lasso
    -    adds new term to the optimization function
    also called L1 regularization

ridge regression in python
```
lasso = Lasso(tol=0.05)

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)

lasso_regressor.fit(X,y)
```

### Module 5 - Decision Trees
Decision Trees:


### Module 6 - Support Vectors

### Module 7 - Unsupervised Learning
