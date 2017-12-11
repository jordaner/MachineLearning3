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

sumFeatures = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5 (meaningless but please still use it)',
            'Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']

whiteWineFeatures = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides']

housePriceFeatures = ['MSSubClass',	'MSZoning',	'LotFrontage',	'LotArea',	'Street',
            'Alley',	'LotShape',	'LandContour',	'Utilities',	'LotConfig',
            'LandSlope',	'Neighborhood',	'Condition1',	'Condition2',	'BldgType',
            'HouseStyle',	'OverallQual',	'OverallCond',
            'YearBuilt',	'YearRemodAdd',	'RoofStyle',	'RoofMatl',
            'Exterior1st',	'Exterior2nd',	'MasVnrType',	'MasVnrArea',	'ExterQual',
            'ExterCond',	'Foundation',	'BsmtQual',	'BsmtCond',	'BsmtExposure',
            'BsmtFinType1',	'BsmtFinSF1',	'BsmtFinType2',	'BsmtFinSF2',
            'BsmtUnfSF',	'TotalBsmtSF',	'Heating',	'HeatingQC',	'CentralAir',
            'Electrical',	'1stFlrSF',	'2ndFlrSF',	'LowQualFinSF',	'GrLivArea',	'BsmtFullBath',
            'BsmtHalfBath',	'FullBath',	'HalfBath',	'BedroomAbvGr',	'KitchenAbvGr',
            'KitchenQual',	'TotRmsAbvGrd',	'Functional',	'Fireplaces',	'FireplaceQu',	'GarageType',
            'GarageYrBlt',	'GarageFinish'	'GarageCars',	'GarageArea',
            'GarageQual',	'GarageCond',	'PavedDrive',	'WoodDeckSF',	'OpenPorchSF',
            'EnclosedPorch',	'3SsnPorch',	'ScreenPorch',	'PoolArea',	'PoolQC',	'Fence',
            'MiscFeature',	'MiscVal',	'MoSold',	'YrSold',	'SaleType',	'SaleCondition']

samples_sizes= [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

le = preprocessing.LabelEncoder()

i = 0
while i<len(samples_sizes):

    sumWithoutNoise = pd.read_csv(“LOCAL PATH”, sep=";", nrows=samples_sizes[i])

    sumWithNoise = pd.read_csv("LOCAL PATH", sep=";",nrows=samples_sizes[i])

    whiteWine = pd.read_csv("LOCAL PATH", sep=";", nrows=samples_sizes[i])

    housePrices = pd.read_csv("LOCAL PATH",sep=",",nrows = samples_sizes[i])


    catnumWithout = sumWithoutNoise["Target Class"].tolist()
    catnumWith =  sumWithNoise["Noisy Target Class"].tolist()

    sumWithoutX = sumWithoutNoise.loc[:, sumFeatures]
    sumWithoutY = sumWithoutNoise.Target
    sumWithoutY_a = le.fit(catnumWithout)
    sumWithoutY_b = le.transform(catnumWithout)

    sumWithX = sumWithNoise.loc[:, sumFeatures]
    sumWithY = sumWithNoise["Noisy Target"]
    sumWithY_a = le.fit(catnumWith)
    sumWithY_b = le.transform(catnumWith)

    wineX = whiteWine.loc[:, whiteWineFeatures]
    wineY = whiteWine.quality

    houseX = housePrices.loc[:, housePriceFeatures]
    houseY = housePrices.SalePrice

    lm = linear_model.LinearRegression(normalize=True)
    rr = linear_model.Ridge(normalize=True)
    lr = linear_model.LogisticRegression()
    nn = neighbors.KNeighborsClassifier(len(samples_sizes))


    kfold = KFold(n_splits=10,random_state=0)

    NMSE_resultsWithoutLR = cross_val_score(lm, sumWithoutX, sumWithoutY, cv=10,scoring="neg_mean_squared_error")  # Choose another regression metric
    NMSE_resultsWithoutLR = NMSE_resultsWithoutLR * -1
    RMS_resultsWithoutLR = np.sqrt(NMSE_resultsWithoutLR)  # LINEAR REGRESSION SUM WITHOUT NOISE
    RMS_resultsWithoutLR = normaliseScores(RMS_resultsWithoutLR)
    mean_errorWithoutLR = RMS_resultsWithoutLR.mean()
    abs_mean_errorWithoutLR = cross_val_score(lm, sumWithoutX, sumWithoutY, cv=10, scoring="neg_mean_absolute_error")
    abs_mean_errorWithoutLR = abs_mean_errorWithoutLR * -1
    abs_mean_errorWithoutLR = normaliseScores(abs_mean_errorWithoutLR)
    abs_mean_errorWithoutLR = abs_mean_errorWithoutLR.mean()

    NMSE_resultsWithoutRR = cross_val_score(rr, sumWithoutX, sumWithoutY, cv=10,scoring="neg_mean_squared_error")  # Choose another regression metric
    NMSE_resultsWithoutRR = NMSE_resultsWithoutRR * -1
    RMS_resultsWithoutRR = np.sqrt(NMSE_resultsWithoutRR)  # LINEAR REGRESSION SUM WITHOUT NOISE
    RMS_resultsWithoutRR = normaliseScores(RMS_resultsWithoutRR)
    mean_errorWithoutRR = RMS_resultsWithoutRR.mean()  # RIDGE REGRESSION SUM WITHOUT NOISE
    abs_mean_errorWithoutRR = cross_val_score(lm, sumWithoutX, sumWithoutY, cv=10, scoring="neg_mean_absolute_error")
    abs_mean_errorWithoutRR = abs_mean_errorWithoutRR * -1
    abs_mean_errorWithoutRR = normaliseScores(abs_mean_errorWithoutRR)
    abs_mean_errorWithoutRR = abs_mean_errorWithoutRR.mean()

    ACC_resultsLogWithout = cross_val_score(lr, sumWithoutX, sumWithoutY_b, cv=kfold, scoring="accuracy")
    PREC_scorerLogWithout = make_scorer(precision_score, average="weighted")
    PREC_resultsLogWithout = cross_val_score(lr, sumWithoutX, sumWithoutY_b, cv=kfold,scoring=PREC_scorerLogWithout)  # LOGISTIC REGRESSION SUM WITHOUT NOISE
    mean_ACCLogWithout = ACC_resultsLogWithout.mean()
    mean_PRECLogWithout = PREC_resultsLogWithout.mean()

    ACC_resultsNnWithout = cross_val_score(nn, sumWithoutX, sumWithoutY_b, cv=kfold, scoring="accuracy")
    PREC_scorerNnWithout = make_scorer(precision_score, average="weighted")
    PREC_resultsNnWithout = cross_val_score(nn, sumWithoutX, sumWithoutY_b, cv=kfold, scoring=PREC_scorerNnWithout)                         #NEAREST NEIGHBORS SUM WITHOUT NOISE
    mean_ACCNnWithout = ACC_resultsNnWithout.mean()
    mean_PRECNnWithout = PREC_resultsNnWithout.mean()

    NMSE_resultsWithLR = cross_val_score(lm, sumWithX, sumWithY_b, cv=10,scoring="neg_mean_squared_error")  # Choose another regression metric
    NMSE_resultsWithLR = NMSE_resultsWithLR * -1
    RMS_resultsWithLR = np.sqrt(NMSE_resultsWithLR)  # LINEAR REGRESSION SUM WITH NOISE
    RMS_resultsWithLR = normaliseScores(RMS_resultsWithLR)
    mean_errorWithLR = RMS_resultsWithLR.mean()
    abs_mean_errorWithLR = cross_val_score(lm, sumWithX, sumWithY_b, cv=10, scoring="neg_mean_absolute_error")
    abs_mean_errorWithLR = abs_mean_errorWithLR * -1
    abs_mean_errorWithLR = normaliseScores(abs_mean_errorWithLR)
    abs_mean_errorWithLR = abs_mean_errorWithLR.mean()

    NMSE_resultsWithRR = cross_val_score(rr, sumWithX, sumWithY, cv=10,scoring="neg_mean_squared_error")  # Choose another regression metric
    NMSE_resultsWithRR = NMSE_resultsWithRR * -1
    RMS_resultsWithRR = np.sqrt(NMSE_resultsWithRR)  # LINEAR REGRESSION SUM WITH NOISE
    RMS_resultsWithRR = normaliseScores(RMS_resultsWithRR)
    mean_errorWithRR = RMS_resultsWithRR.mean()
    abs_mean_errorWithRR = cross_val_score(rr, sumWithX, sumWithY, cv=10,scoring="neg_mean_absolute_error")  # RIDGE REGRESSION SUM WITH NOISE
    abs_mean_errorWithRR = abs_mean_errorWithRR * -1
    abs_mean_errorWithRR = normaliseScores(abs_mean_errorWithRR)
    abs_mean_errorWithRR = abs_mean_errorWithRR.mean()

    ACC_resultsLogWith = cross_val_score(lr, sumWithX, sumWithY_b, cv=kfold, scoring="accuracy")
    PREC_scorerLogWith = make_scorer(precision_score, average="weighted")
    PREC_resultsLogWith = cross_val_score(lr, sumWithX, sumWithY_b, cv=kfold,scoring=PREC_scorerLogWith)  # LOGISTIC REGRESSION SUM With NOISE
    mean_ACCLogWith = ACC_resultsLogWith.mean()
    mean_PRECLogWith = PREC_resultsLogWith.mean()

    ACC_resultsNnWith = cross_val_score(nn, sumWithX, sumWithY_b, cv=kfold, scoring="accuracy")
    PREC_scorerNnWith = make_scorer(precision_score, average="weighted")
    PREC_resultsNnWith = cross_val_score(nn, sumWithX, sumWithY_b, cv=kfold,scoring=PREC_scorerNnWith)  # NEAREST NEIGHBORS SUM WITH NOISE
    mean_ACCNnWith = ACC_resultsNnWith.mean()
    mean_PRECNnWith = PREC_resultsNnWith.mean()

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
    housePrices["target_class"] = encoder.fit_transform(housePrices["SalePrice"].tolist())

    for column in houseX:
        if "object" in str(houseX[column].dtype):
            houseX[column] = transformColumn(houseX[column])
    houseX = houseX.replace(np.nan,0)

    logHouseY = [transformValueToClassValue(i) for i in (houseY.tolist())]
    logHouseY = pd.Series(data=logHouseY)

    NMSE_resultsHouseLR = cross_val_score(lm, houseX, houseY, cv=10,scoring="neg_mean_squared_error")  # Choose another regression metric
    NMSE_resultsHouseLR = NMSE_resultsHouseLR * -1
    RMS_resultsHouseLR = np.sqrt(NMSE_resultsHouseLR)
    RMS_resultsHouseLR = normaliseScores(RMS_resultsHouseLR)
    mean_errorHouseLR = RMS_resultsHouseLR.mean()
    abs_mean_errorHouseLR = cross_val_score(lm, sumWithoutX, sumWithoutY, cv=10,scoring="neg_mean_absolute_error")  # LINEAR REGRESSION HOUSE PRICES DATA SETabs_mean_errorHouseLR = cross_val_score(lm, houseX, houseY, cv=10, scoring="neg_mean_absolute_error")
    abs_mean_errorHouseLR = abs_mean_errorHouseLR * -1
    abs_mean_errorHouseLR = normaliseScores(abs_mean_errorHouseLR)
    abs_mean_errorHouseLR = abs_mean_errorHouseLR.mean()

    NMSE_resultsHouseRR = cross_val_score(rr, houseX, houseY, cv=10,scoring="neg_mean_squared_error")  # Choose another regression metric
    NMSE_resultsHouseRR = NMSE_resultsHouseRR * -1
    RMS_resultsHouseRR = np.sqrt(NMSE_resultsHouseRR)
    RMS_resultsHouseRR = normaliseScores(RMS_resultsHouseRR)
    mean_errorHouseRR = RMS_resultsHouseRR.mean()  # RIDGE REGRESSION HOUSE PRICES DATA SET
    abs_mean_errorHouseRR = cross_val_score(rr, houseX, houseY, cv=10, scoring="neg_mean_absolute_error")
    abs_mean_errorHouseRR = abs_mean_errorHouseRR * -1
    abs_mean_errorHouseRR = normaliseScores(abs_mean_errorHouseRR)
    abs_mean_errorHouseRR = abs_mean_errorHouseRR.mean()

    ACC_resultsLogHouse = cross_val_score(lr, houseX, logHouseY, cv=kfold, scoring="accuracy")
    PREC_scorerLogHouse = make_scorer(precision_score, average="weighted")
    PREC_resultsLogHouse = cross_val_score(lr, houseX, logHouseY, cv=kfold, scoring=PREC_scorerLogHouse)                #LOGISTIC REGRESSION HOUSE PRICES DATA SET
    mean_ACCLogHouse = ACC_resultsLogHouse.mean()
    mean_PRECLogHouse = PREC_resultsLogHouse.mean()

    ACC_resultsNnHouse = cross_val_score(nn, houseX, logHouseY, cv=kfold, scoring="accuracy")
    PREC_scorerNnHouse = make_scorer(precision_score, average="weighted")
    PREC_resultsNnHouse = cross_val_score(nn, houseX, logHouseY, cv=kfold, scoring=PREC_scorerNnHouse)                  #NEAREST NEIGHBORS HOUSE PRICES DATA SET
    mean_ACCNnHouse = ACC_resultsNnHouse.mean()
    mean_PRECNnHouse = PREC_resultsNnHouse.mean()

    print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_errorHouseLR,
          "for the house prices data set - Linear Regression")
    print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_errorHouseLR,
          "for the house prices data set - Linear Regression")

    print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_errorHouseLR,
          "for the house prices data set - Ridge Regression")
    print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_errorHouseLR,
          "for the house prices data set - Ridge Regression")

    print("Accuracy with sample of size of ", samples_sizes[i], " = ", mean_ACCLogHouse,
          " for the House Prices Data Set - Logstic Regression")
    print("Precision Score with sample of size of ", samples_sizes[i], " = ", mean_PRECLogHouse,
          "for the House Prices Data Set - Logistic Regression")

    print("Accuracy with sample of size of ", samples_sizes[i], " = ", mean_ACCNnHouse,
          " for the White Wine Data Set - Nearest Neighbors")
    print("Precision Score with sample of size of ", samples_sizes[i], " = ", mean_PRECNnHouse,
          " for the White Wine Data Set - Nearest Neighbors")

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

    print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_errorWithLR,
          "for SUM with Noise - Linear Regression")
    print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_errorWithLR,
          "for SUM with Noise - Linear Regression")

    print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_errorWithRR,
          "for SUM with Noise - Ridge Regression")
    print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_errorWithRR,
          "for SUM with Noise - Ridge Regression")

    print("Accuracy with sample of size of ", samples_sizes[i], " = ", mean_ACCLogWith,
          " for SUM without noise - Logistic Regression")
    print("Precision Score with sample of size of ", samples_sizes[i], " = ", mean_PRECLogWith,
          "for SUM without noise - Logistic Regression")

    print(
        "Accuracy with sample of size of ", samples_sizes[i], " = ", mean_ACCNnWith,
        " for SUM with noise - Nearest Neighbors")
    print("Precision Score with sample of size of ", samples_sizes[i], " = ", mean_PRECNnWith,
          " for SUM with noise - Nearest Neighbors")

    print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_errorWithoutLR,
          "for SUM without Noise - Linear Regression")
    print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_errorWithoutLR,
          "for SUM without Noise - Linear Regression")

    print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_errorWithoutRR,
          "for SUM without Noise - Ridge Regression")
    print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_errorWithoutRR,
          "for SUM without Noise - Ridge Regression")

    print("Accuracy with sample of size of ", samples_sizes[i], " = ", mean_ACCLogWithout,
          " for SUM without noise - Logistic Regression")
    print("Precision Score with sample of size of ", samples_sizes[i], " = ", mean_PRECLogWithout,
          "for SUM without noise - Logistic Regression")

    print("Accuracy with sample of size of ", samples_sizes[i], " = ", mean_ACCNnWithout,
          " for SUM without noise - Nearest Neighbors")
    print("Precision Score with sample of size of ", samples_sizes[i], " = ", mean_PRECNnWithout,
          " for SUM without noise - Nearest Neighbors")

    i += 1
