import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.metrics import mean_absolute_error, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing, neighbors
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def transformColumn(column):
    transformed_list = LabelEncoder().fit_transform(column.tolist())
    transformed_series = pd.Series(data=transformed_list)
    transformed_series = pd.Series(data=transformed_list)
    transformed_series.replace(np.NaN, 0)
    transformed_series.set_value(100, 2)
    return transformed_series

def transformValueToClassValue(value):
    if "str" in str(type(value)):
        return value
    else:
        return round(value/100000)

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

def featureSelect(X, Y):
    X_new = SelectKBest(chi2, k=4).fit_transform(X,Y)
    return X_new


whiteWineFeatures = ['fixed acidity','volatile acidity','citric acid',
                     'residual sugar','chlorides','free sulfur dioxide',
                     'total sulfur dioxide','density','pH','sulphates','alcohol'];

le = preprocessing.LabelEncoder()

i = 0


whiteWine = pd.read_csv("/Users/markloughman/Desktop/winequality-white.csv", sep=";")

whiteX = whiteWine.loc[:, whiteWineFeatures]
whiteY = whiteWine.quality

whiteX = featureSelect(whiteX, whiteY);

whiteX = scaleData(whiteX);

lm = linear_model.LinearRegression(normalize=True)
rr = linear_model.Ridge(normalize=True)
lr = linear_model.LogisticRegression()
nn = neighbors.KNeighborsClassifier()

kfold = KFold(n_splits=10,random_state=0)

NMSE_resultsWhiteWineLR = cross_val_score(lm, whiteX, whiteY, cv=10, scoring="neg_mean_squared_error")  # Choose another regression metric
NMSE_resultsWhiteWineLR = NMSE_resultsWhiteWineLR * -1
RMS_resultsWhiteWineLR = np.sqrt(NMSE_resultsWhiteWineLR)
RMS_resultsWhiteWineLR = normaliseScores(RMS_resultsWhiteWineLR)  # LINEAR REGRESSION WHITE WINE DATA SET
mean_errorWhiteWineLR = RMS_resultsWhiteWineLR.mean()
abs_mean_errorWhiteWineLR = cross_val_score(lm, whiteX, whiteY, cv=10, scoring="neg_mean_absolute_error")
abs_mean_errorWhiteWineLR = abs_mean_errorWhiteWineLR * -1
abs_mean_errorWhiteWineLR = normaliseScores(abs_mean_errorWhiteWineLR)
abs_mean_errorWhiteWineLR = abs_mean_errorWhiteWineLR.mean()

NMSE_resultsWhiteWineRR = cross_val_score(rr, whiteX, whiteY, cv=10,scoring="neg_mean_squared_error")  # Choose another regression metric
NMSE_resultsWhiteWineRR = NMSE_resultsWhiteWineRR * -1
RMS_resultsWhiteWineRR = np.sqrt(NMSE_resultsWhiteWineRR)
RMS_resultsWhiteWineRR = normaliseScores(RMS_resultsWhiteWineRR)  # RIDGE REGRESSION WHITE WINE DATA SET
mean_errorWhiteWineRR = RMS_resultsWhiteWineRR.mean()
abs_mean_errorWhiteWineRR = cross_val_score(rr, whiteX, whiteY, cv=10, scoring="neg_mean_absolute_error")
abs_mean_errorWhiteWineRR = abs_mean_errorWhiteWineRR * -1
abs_mean_errorWhiteWineRR = normaliseScores(abs_mean_errorWhiteWineRR)
abs_mean_errorWhiteWineRR = abs_mean_errorWhiteWineRR.mean()

ACC_resultsLogWhiteWine = cross_val_score(lr, whiteX, whiteY, cv=kfold, scoring="accuracy")
PREC_scorerLogWhiteWine = make_scorer(precision_score, average="weighted")
PREC_resultsLogWhiteWine = cross_val_score(lr, whiteX, whiteY, cv=kfold, scoring=PREC_scorerLogWhiteWine)  # LOGISTIC REGRESSION SUM WhiteWine NOISE
mean_ACCLogWhiteWine = ACC_resultsLogWhiteWine.mean()
mean_PRECLogWhiteWine = PREC_resultsLogWhiteWine.mean()

ACC_resultsNnWhiteWine = cross_val_score(nn, whiteX, whiteY, cv=kfold, scoring="accuracy")
PREC_scorerNnWhiteWine = make_scorer(precision_score, average="weighted")
PREC_resultsNnWhiteWine = cross_val_score(nn, whiteX, whiteY, cv=kfold,scoring=PREC_scorerNnWhiteWine)  # NEAREST NEIGHBORS WHITE WINE DATA SET
mean_ACCNnWhiteWine = ACC_resultsNnWhiteWine.mean()
mean_PRECNnWhiteWine = PREC_resultsNnWhiteWine.mean()

encoder = LabelEncoder()


print("Error with sample size of for mean squared error = ", mean_errorWhiteWineLR,
          "for the White Wine Data Set - Linear Regression")
print("Error with sample size of for absolute mean error =", abs_mean_errorWhiteWineLR,
          "for the White Wine Data Set - Linear Regression")


print("Error with sample size of for mean squared error = ", mean_errorWhiteWineRR,
          "for the White Wine Data Set - Ridge Regression")
print("Error with sample size of for absolute mean error =", abs_mean_errorWhiteWineRR,
          "for the White Wine Data Set - Ridge Regression")

print("Accuracy with sample of size of = ", mean_ACCLogWhiteWine,
          " for the White Wine Data Set - Logstic Regression")
print("Precision Score with sample of size of = ", mean_PRECLogWhiteWine,
          "for the White Wine Data Set - Logistic Regression")

print("Accuracy with sample of size of = ", mean_ACCNnWhiteWine,
          " for the White Wine Data Set - Nearest Neighbors")
print("Precision Score with sample of size of = ", mean_PRECNnWhiteWine,
          " for the White Wine Data Set - Nearest Neighbors")

