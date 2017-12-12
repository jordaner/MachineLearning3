import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import scipy as sp
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.metrics import mean_absolute_error, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing, neighbors
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import warnings

def normaliseScores(scores):
    old_max = max(scores)
    old_min = min(scores)
    old_range = old_max - old_min
    new_min = 0
    new_max = 1
    normalised_scores = np.array([(new_min + (((x-old_min)*(new_max-new_min)))/(old_max - old_min)) for x in scores])
    return normalised_scores

def scaleData(unscaledData):
    scaledData = preprocessing.scale(unscaledData)
    return scaledData;

def featureSelect(X, Y, i):
    X_new = SelectKBest(chi2, k=i).fit_transform(X,Y)
    return X_new

def executeAlgorithms(X, y):
    # These won't work as the evaluation metris are set up for classification not regression
    print("--- LinearRegression ---")
    model = linear_model.LinearRegression(normalize=True)
    model = model.fit(X, y)
    regMetricEvaluation(model, X, y)

    print("--- Ridge Regression ---")
    model = linear_model.Ridge(normalize = True)
    model = model.fit(X, y)
    regMetricEvaluation(model, X, y)

    print("--- Nearest Neighbors ---")
    model = neighbors.KNeighborsClassifier()
    model = model.fit(X, y)
    classMetricEvaluation(model, X, y)

    print("--- Logistic Regression ---")
    model = linear_model.LogisticRegression()
    model = model.fit(X, y)
    classMetricEvaluation(model, X, y)

    return

def regMetricEvaluation(model, X, y):
    # RMS Error
    mean_squared_error = cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_error") * -1
    mean_squared_error = normaliseScores(mean_squared_error)
    root_mean_squared_error = np.sqrt(mean_squared_error)
    # Absoulte mean error
    abs_mean_error = cross_val_score(model, X, y, cv=10, scoring="neg_mean_absolute_error") * -1
    abs_mean_error = normaliseScores(abs_mean_error)
    # R2 score
    r2_score = cross_val_score(model, X, y, cv=10, scoring="r2")
    r2_score = normaliseScores(r2_score)
    # Median absolute error
    median_absolute_error = cross_val_score(model, X, y, cv=10, scoring="neg_median_absolute_error") * -1
    median_absolute_error = normaliseScores(median_absolute_error)
    # Mean squared log error
    mean_squared_log_error = cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_log_error") * -1
    mean_squared_log_error = normaliseScores(mean_squared_log_error)

    # Runtime metric
    # start_time = time.time()
    # cross_val_score(model, X, y, cv=10)
    # runtime = time.time() - start_time
    # print("Runtime                =", runtime)
    print("Mean squared log error            =", mean_squared_log_error.mean())
    print("Median absolute error             =", median_absolute_error.mean())
    print("R2 score                          =", r2_score.mean())
    print("RMS error                         =", root_mean_squared_error.mean())
    print("Absolute mean error               =", abs_mean_error.mean())
    # print("Absolute mean error per unit time =", (abs_mean_error.mean()/runtime))

    # times.insert(0, runtime)
    # errorPerUnitTime.insert(0, abs_mean_error.mean()/runtime)

    return

def classMetricEvaluation(model, X, y):
    accuracy_result = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    f1_weighted_result = cross_val_score(model, X, y, cv=10, scoring="f1_weighted")
    f1_micro_result = cross_val_score(model, X, y, cv=10, scoring="f1_micro")
    f1_macro_result = cross_val_score(model, X, y, cv=10, scoring="f1_macro")
    # f1_samples_result = cross_val_score(model, X, y, cv=10, scoring="f1_samples")
    precision_weighted_result = cross_val_score(model, X, y, cv=10, scoring="precision_weighted")
    # average_precision_result = cross_val_score(model, X, y, cv=10, scoring="average_precision")
    # TODO Replace precision with F1_score
    # f1_scorer = make_scorer(f1_score, average="weighted")
    # 'f1_score' is not a valid scoring value. Valid options are ['accuracy', 'adjusted_mutual_info_score',
    # 'adjusted_rand_score', 'average_precision', 'completeness_score', 'explained_variance', 'f1', 'f1_macro',
    # 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'mutual_info_score',
    # 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error',
    # 'neg_median_absolute_error', 'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro',
    # 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples',
    #  'recall_weighted', 'roc_auc', 'v_measure_score']
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
    print("Accuracy                         =", accuracy_result.mean(), "%")
    print("F1_weighted                      =", f1_weighted_result.mean(),"%")
    print("Precision_weighted               =", precision_weighted_result.mean(),"%")
    print("F1_micro                         =", f1_micro_result.mean(),"%")
    print("F1_macro                         =", f1_macro_result.mean(),"%")
    # print("F1_samples                      =", f1_samples_result.mean(),"%")
    # print("Precision_average                      =", average_precision_result.mean(),"%")
    # print("Median absolute error             =", median_absolute_error.mean())
    # print("R2 score                          =", r2_score.mean())
    # print("Absolute mean error               =", abs_mean_error.mean())
    # print("Absolute mean error per unit time =", (abs_mean_error.mean()/runtime))

    # times.insert(0, runtime)
    # errorPerUnitTime.insert(0, abs_mean_error.mean()/runtime)

    return

### Added to suppress warnings for ill-defined precision, should be removed if other issues arise
#warnings.filterwarnings("ignore")

Features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
#
# whiteWine = pd.read_csv("~/Desktop/Python/ML3/", sep=";")
# redWine = pd.read_csv("/Users/markloughman/Desktop/winequality-white.csv", sep=";")

print("----- winequality-red -----")
dataframe = pd.read_csv("/home/eric/Desktop/4th Year/MachineLearning/Assignment3/Wine/winequality-white.csv", sep=";")
X = dataframe.loc[:, Features]
y = dataframe.quality

# clf = neighbors.LocalOutlierFactor(n_neighbors=20)
# y_pred = clf.fit_predict(X)
# y_pred_outliers = y_pred
# count = 0
# for i in range(len(y_pred_outliers)):
#     if y_pred_outliers[i] == -1:
#         count += 1
# print(count)
# print(len(y)-count)

i = 11

while i > 0 :
    Xs = featureSelect(X, y, i);
    Xs = scaleData(Xs);

    print("------ Training with",i,"features ------")
    executeAlgorithms(Xs, y)
    i -= 1
