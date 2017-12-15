import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import scipy as sp
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.metrics import mean_absolute_error, precision_score, f1_score,median_absolute_error,mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing, neighbors
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import warnings
import math
import sys
from sklearn import svm

def scaleData(unscaledData):
    scaledData = preprocessing.scale(unscaledData)
    return scaledData

def featureSelect(X, Y, i):
    X_new = SelectKBest(chi2, k=i).fit_transform(X,Y)
    return X_new

def executeRegression(X, y):
    print("--- LinearRegression ---")
    model = linear_model.LinearRegression()
    regMetricEvaluation(model, X, y)

    print("--- Ridge Regression ---")
    model = linear_model.Ridge()
    regMetricEvaluation(model, X, y)
    return

def regMetricEvaluation(model, X, y):
    # RMS Error
    mean_squared_error = cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_error") * -1
    root_mean_squared_error = np.sqrt(mean_squared_error)
    # Absoulte mean error
    abs_mean_error = cross_val_score(model, X, y, cv=10, scoring="neg_mean_absolute_error") * -1
    # R2 score
    r2_score = cross_val_score(model, X, y, cv=10, scoring="r2")
    # Median absolute error
    median_absolute_error = cross_val_score(model, X, y, cv=10, scoring="neg_median_absolute_error") * -1
    # Mean squared log error
    mean_squared_log_error = cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_log_error") * -1

    print(math.sqrt(mean_squared_log_error.mean()))
    print(median_absolute_error.mean())
    print(r2_score.mean())
    print(root_mean_squared_error.mean())
    print(abs_mean_error.mean())
    return

# Outputting results into a file called results.txt
f = open("results.txt", 'w')
sys.stdout = f

print("----- winequality-white -----")
Features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
            'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

df = pd.read_csv("/home/eric/Desktop/4th Year/MachineLearning/Assignment3/Wine/winequality-red.csv", sep=";")
X = df.loc[:, Features]
y = df.quality
outliers_fraction = 0.01
nu_estimate = 0.95 * outliers_fraction + 0.05
auto_detection = svm.OneClassSVM(kernel = "rbf", gamma = 0.01, degree = 3, nu = nu_estimate)
auto_detection.fit(X)
evaluation = auto_detection.predict(X)
dataframe = df[evaluation==1]
X2 = dataframe.loc[:, Features]
y2 = dataframe.quality

i = 11
while i > 0 :
    Xs = featureSelect(X, y, i)
    Xs = scaleData(Xs)

    Xs2 = featureSelect(X2, y2, i)
    Xs2 = scaleData(Xs2)

    print("------ Training with",i,"features ------")
    print("----- Cross_Val Regression with outliers -----")
    executeRegression(Xs, y)
    print("----- Cross_Val Regression outliers removed -----")
    executeRegression(Xs2, y2)
    i -= 1
