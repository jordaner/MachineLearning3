import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import scipy as sp
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.metrics import mean_absolute_error, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing, neighbors
from sklearn.metrics import accuracy_score

def normaliseScores(scores):
    old_max = max(scores)
    old_min = min(scores)
    old_range = old_max - old_min
    new_min = 0
    new_max = 1
    normalised_scores = np.array([(new_min + (((x-old_min)*(new_max-new_min)))/(old_max - old_min)) for x in scores])
    return normalised_scores

def executeAlgorithms(X, y):
    # These won't work as the evaluation metris are set up for classification not regression
    # print("--- LinearRegression ---")
    # model = linear_model.LinearRegression(normalize=True)
    # model = model.fit(X, y)
    # metricEvaluation(model, X, y)
    #
    # print("--- Ridge Regression ---")
    # model = linear_model.Ridge(normalize = True)
    # model = model.fit(X, y)
    # metricEvaluation(model, X, y)
    #
    print("--- Nearest Neighbors ---")
    model = neighbors.KNeighborsClassifier(5)
    model = model.fit(X, y)
    metricEvaluation(model, X, y)

    print("--- Logistic Regression ---")
    model = linear_model.LogisticRegression()
    model = model.fit(X, y)
    metricEvaluation(model, X, y)

    return

def metricEvaluation(model, X, y):
    accuracy = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    # precision_scorer = make_scorer(precision_score, average="weighted")
    # precision = cross_val_score(model, X, y, cv=10, scoring=precision_scorer)
    # RMS Error
    # mean_squared_error = cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_error") * -1
    # mean_squared_error = normaliseScores(mean_squared_error)
    # root_mean_squared_error = np.sqrt(mean_squared_error)
    # Absoulte mean error
    # abs_mean_error = cross_val_score(model, X, y, cv=10, scoring="neg_mean_absolute_error")
    # abs_mean_error = abs_mean_error * -1
    # abs_mean_error = normaliseScores(abs_mean_error)
    # R2 score
    # r2_score = cross_val_score(model, X, y, cv=10, scoring="r2")
    # r2_score = normaliseScores(r2_score)
    # Median absolute error
    # median_absolute_error = cross_val_score(model, X, y, cv=10, scoring="neg_median_absolute_error") * -1
    # median_absolute_error = normaliseScores(median_absolute_error)
    # Mean squared log error
    # mean_squared_log_error = cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_log_error") * -1
    # mean_squared_log_error = normaliseScores(mean_squared_log_error)# times.insert(0, runtime)
    # errorPerUnitTime.insert(0, abs_mean_error.mean()/runtime)

    # Runtime metric
    # start_time = time.time()
    # cross_val_score(model, X, y, cv=10)
    # runtime = time.time() - start_time
    print("Accuracy                         =", accuracy.mean(),"%")
    # print("Precision                        =", precision.mean(),"%")
    # print("Median absolute error             =", median_absolute_error.mean())
    # print("R2 score                          =", r2_score.mean())
    # print("Absolute mean error               =", abs_mean_error.mean())
    # print("Absolute mean error per unit time =", (abs_mean_error.mean()/runtime))

    # times.insert(0, runtime)
    # errorPerUnitTime.insert(0, abs_mean_error.mean()/runtime)

    return

Features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
#
# whiteWine = pd.read_csv("~/Desktop/Python/ML3/", sep=";")
# redWine = pd.read_csv("/Users/markloughman/Desktop/winequality-white.csv", sep=";")

print("----- winequality-red -----")
dataframe = pd.read_csv("~/Desktop/Python/winequality-red.csv", sep=";")
X = dataframe.loc[:, Features]
y = dataframe.quality

executeAlgorithms(X, y)
