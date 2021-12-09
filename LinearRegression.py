import pandas as pd
import numpy as np
import math
from scipy import stats
from tabulate import tabulate


def linearRegression(X, y):

    # transforming the inputs into numpy arrays and concatenating them
    X = np.array(X)
    y = np.array(y)
    t = np.c_[X, y]

    # list-wise deletion for handling NaN values and seperating X and y back
    t = t[~np.isnan(t).any(axis=1), :]
    X = np.array(t[:, :-1])
    y = np.array(t[:, -1:])

    # adding a column of 1's to X for b0
    X = np.c_[np.ones(len(X)), X]

    # setting n and k values
    n = len(X)
    k = len(X[0])

    # performing linear regression algorithm
    betaHeadMatrix = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), y)

    yPredictions = np.matmul(X, betaHeadMatrix)

    residuals = y - yPredictions

    sigmaSquare = np.matmul(residuals.transpose(), residuals) / (n - k - 1)

    varBetaHead = sigmaSquare * np.linalg.inv(np.matmul(X.transpose(), X))

    standardErrors = []
    for i in range(len(X[0])):
        standardErrors.append(math.sqrt(varBetaHead[i][i]))

    confidanceIntervals = []
    counter = 0
    for betaHead in betaHeadMatrix:
        confidanceIntervals.append([betaHead[0] - stats.t.ppf(1 - 0.025, n - k - 1) * standardErrors[counter],
                                    betaHead[0] + stats.t.ppf(1 - 0.025, n - k - 1) * standardErrors[counter]])
        counter += 1

    # tabulating the results and printing them
    table = [['Estimator', 'Coefficient', 'Standard Error', 'Lower 95%', 'Upper 95%']]
    i = 0
    for betaHead in betaHeadMatrix:
        table.append(
            ['b' + str(i), betaHead[0], standardErrors[i], confidanceIntervals[i][0], confidanceIntervals[i][1]])
        i += 1
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    return betaHeadMatrix, standardErrors, confidanceIntervals
