import matplotlib.pyplot as plt
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

whiteWineFeatures = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides']

samples_sizes= [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

le = preprocessing.LabelEncoder()

i = 0
while i<len(samples_sizes):

    whiteWine = pd.read_csv("/Users/markloughman/Desktop/winequality-white.csv", sep=";", nrows=samples_sizes[i])

    wineX = whiteWine.loc[:, whiteWineFeatures]
    wineY = whiteWine.quality

    lm = linear_model.LinearRegression(normalize=True)
    rr = linear_model.Ridge(normalize=True)
    lr = linear_model.LogisticRegression()
    nn = neighbors.KNeighborsClassifier(len(samples_sizes))

    kfold = KFold(n_splits=10,random_state=0)

    NMSE_resultsWhiteWineLR = cross_val_score(lm, wineX, wineY, cv=10,scoring="neg_mean_squared_error")  # Choose another regression metric
    NMSE_resultsWhiteWineLR = NMSE_resultsWhiteWineLR * -1
    RMS_resultsWhiteWineLR = np.sqrt(NMSE_resultsWhiteWineLR)
    RMS_resultsWhiteWineLR = normaliseScores(RMS_resultsWhiteWineLR)  # LINEAR REGRESSION WHITE WINE DATA SET
    mean_errorWhiteWineLR = RMS_resultsWhiteWineLR.mean()
    abs_mean_errorWhiteWineLR = cross_val_score(lm, wineX, wineY, cv=10, scoring="neg_mean_absolute_error")
    abs_mean_errorWhiteWineLR = abs_mean_errorWhiteWineLR * -1
    abs_mean_errorWhiteWineLR = normaliseScores(abs_mean_errorWhiteWineLR)
    abs_mean_errorWhiteWineLR = abs_mean_errorWhiteWineLR.mean()

    NMSE_resultsWhiteWineRR = cross_val_score(rr, wineX, wineY, cv=10,scoring="neg_mean_squared_error")  # Choose another regression metric
    NMSE_resultsWhiteWineRR = NMSE_resultsWhiteWineRR * -1
    RMS_resultsWhiteWineRR = np.sqrt(NMSE_resultsWhiteWineRR)
    RMS_resultsWhiteWineRR = normaliseScores(RMS_resultsWhiteWineRR)  # RIDGE REGRESSION WHITE WINE DATA SET
    mean_errorWhiteWineRR = RMS_resultsWhiteWineRR.mean()
    abs_mean_errorWhiteWineRR = cross_val_score(rr, wineX, wineY, cv=10, scoring="neg_mean_absolute_error")
    abs_mean_errorWhiteWineRR = abs_mean_errorWhiteWineRR * -1
    abs_mean_errorWhiteWineRR = normaliseScores(abs_mean_errorWhiteWineRR)
    abs_mean_errorWhiteWineRR = abs_mean_errorWhiteWineRR.mean()

    ACC_resultsLogWhiteWine = cross_val_score(lr, wineX, wineY, cv=kfold, scoring="accuracy")
    PREC_scorerLogWhiteWine = make_scorer(precision_score, average="weighted")
    PREC_resultsLogWhiteWine = cross_val_score(lr, wineX, wineY, cv=kfold, scoring=PREC_scorerLogWhiteWine)  # LOGISTIC REGRESSION SUM WhiteWine NOISE
    mean_ACCLogWhiteWine = ACC_resultsLogWhiteWine.mean()
    mean_PRECLogWhiteWine = PREC_resultsLogWhiteWine.mean()

    ACC_resultsNnWhiteWine = cross_val_score(nn, wineX, wineY, cv=kfold, scoring="accuracy")
    PREC_scorerNnWhiteWine = make_scorer(precision_score, average="weighted")
    PREC_resultsNnWhiteWine = cross_val_score(nn, wineX, wineY, cv=kfold,scoring=PREC_scorerNnWhiteWine)  # NEAREST NEIGHBORS WHITE WINE DATA SET
    mean_ACCNnWhiteWine = ACC_resultsNnWhiteWine.mean()
    mean_PRECNnWhiteWine = PREC_resultsNnWhiteWine.mean()

    encoder = LabelEncoder()


    print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_errorWhiteWineLR,
          "for the White Wine Data Set - Linear Regression")
    print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_errorWhiteWineLR,
          "for the White Wine Data Set - Linear Regression")

    print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_errorWhiteWineRR,
          "for the White Wine Data Set - Ridge Regression")
    print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_errorWhiteWineRR,
          "for the White Wine Data Set - Ridge Regression")

    print("Accuracy with sample of size of ", samples_sizes[i], " = ", mean_ACCLogWhiteWine,
          " for the White Wine Data Set - Logstic Regression")
    print("Precision Score with sample of size of ", samples_sizes[i], " = ", mean_PRECLogWith,
          "for the White Wine Data Set - Logistic Regression")

    print("Accuracy with sample of size of ", samples_sizes[i], " = ", mean_ACCNnWhiteWine,
          " for the White Wine Data Set - Nearest Neighbors")
    print("Precision Score with sample of size of ", samples_sizes[i], " = ", mean_PRECNnWhiteWine,
          " for the White Wine Data Set - Nearest Neighbors")

    i += 1
