import csv
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd

# reading our data from the csv file
data = []
with open("cses4_cut.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)
data = np.asarray(data)
# splitting data into X and y
X = data[:, 1:-1]
y = data[:, -1:]


# FEATURE ENGINEERING

# extracting D2002, D2004, (D2005, D2006, D2007, D2008, D2009), (D2012, D2013, D2014), (D2017, D2018, D2019), D2031
categories = ["D2002", "D2004", "D2005", "D2006", "D2007", "D2008", "D2009", "D2012", "D2013", "D2014", "D2017",
              "D2018", "D2019", "D2031"]
X1 = X[1:, [0, 2, 3, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17, 29]].astype(np.int)
# replacing missing values with the most frequent answer in that category
imp = SimpleImputer(missing_values=9, strategy='most_frequent')
X1 = imp.fit_transform(X1)
# one hat encoding
encoder = OneHotEncoder(sparse=False, dtype=int)
X1 = encoder.fit_transform(X1)
# transforming into dataframe
df1 = pd.DataFrame(X1, index=range(len(X1)), columns=encoder.get_feature_names(categories).tolist())

# extracting D2003,/ D2010, D2015, // D2021, D2022, D2023
categories = ["D2003", "D2010", "D2015", "D2021", "D2022", "D2023"]
X2 = X[1:, [1, 8, 13, 19, 20, 21]].astype(np.int)
# replacing missing values with the most frequent answer in that category
imp = SimpleImputer(missing_values=99, strategy='most_frequent')
X2 = imp.fit_transform(X2)
# transforming into dataframe
df2 = pd.DataFrame(X2, index=range(len(X2)), columns=categories)

# extracting D2020, // D2024, D2025
categories = ["D2020", "D2024", "D2025"]
X3 = X[1:, [18, 22, 23]].astype(np.int)
# replacing missing values with the most frequent answer in that category
imp = SimpleImputer(missing_values=9, strategy='most_frequent')
X3 = imp.fit_transform(X3)
# transforming into dataframe
df3 = pd.DataFrame(X3, index=range(len(X3)), columns=categories)

# gathering all the data together
df = pd.concat([df1, df2, df3], axis=1)


# SPLITTING THE DATA INTO 2 BEFORE TRAINING ANY ALGORITHMS ON THEM
from sklearn.model_selection import train_test_split
Xftrain, Xftest, yftrain, yftest = train_test_split(df, y[1:, :].ravel(), random_state=0)
df = Xftrain
y = yftrain


# TESTING AN EXAMPLE MODEL OF PCA AND GAUSSIAN NAIVE BAYES CLASSIFIER

print("GAUSSIAN NAIVE BAYES CLASSIFIER")
# PCA
from sklearn.decomposition import PCA
model = PCA(n_components=10)
model.fit(df)
X_10D = model.transform(df)
# classification with Gaussian Naive Bayes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
Xtrain, Xtest, ytrain, ytest = train_test_split(X_10D, y, random_state=0)
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
# calculating and printing the accuracy score of the model with simple cross validation
from sklearn.metrics import accuracy_score
score = accuracy_score(ytest, y_model)
print("The accuracy score of the example model is ", score)
# plotting the confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
mat = confusion_matrix(ytest, y_model)
disp = ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
plt.show()
# calculating and printing the accuracy score of the model with 7-fold cross validation
from sklearn.model_selection import cross_val_score
print("7-fold Cross Validation Score List: ")
scoreList = cross_val_score(model, X_10D, y, cv=7)
print(scoreList)
print("Average: ", sum(scoreList) / len(scoreList))
# calculating and printing the accuracy score of the model on test data that model has never seen before
model = PCA(n_components=10)
model.fit(Xftest)
Xftest_10D = model.transform(Xftest)
model = GaussianNB()
model.fit(X_10D, y)
y_model = model.predict(Xftest_10D)
score = accuracy_score(yftest, y_model)
print("Accuracy score on test data: ", score)
mat = confusion_matrix(yftest, y_model)
disp = ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
plt.show()
print("--------------------------------------------")


# FINDING THE OPTIMAL MODEL WITH PCA AND NAIVE BAYES CLASSIFIER

print("THE OPTIMAL MODEL WITH PCA AND NAIVE BAYES CLASSIFIER")
# making the pipeline
from sklearn.pipeline import make_pipeline

def NaiveBayesClassifier(n_components=3, dist=GaussianNB()):
    return make_pipeline(PCA(n_components), dist)

# tuning the best parameters for both PCA and Naive Bayes Classifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
param_grid = {'pca__n_components': np.arange(2, 40), 'gaussiannb': [GaussianNB(), BernoulliNB()]}
grid = GridSearchCV(NaiveBayesClassifier(), param_grid, cv=7)
grid.fit(df, y)

# printing the best parameters
print("The best parameters are:")
print(grid.best_params_)

# printing and setting the model to best estimator
best_model = grid.best_estimator_
print("The best model is: ")
print(best_model)

# calculating and printing the accuracy score of the model with 7-fold cross validation
print("7-fold Cross Validation Score List: ")
scoreList = cross_val_score(best_model, df, y, cv=7)
print(scoreList)
print("Average: ", sum(scoreList) / len(scoreList))

# calculating and printing the accuracy score of the model on test data that model has never seen before
best_model.fit(df, y)
y_model = best_model.predict(Xftest)
score = accuracy_score(yftest, y_model)
print("Accuracy score on test data: ", score)
mat = confusion_matrix(yftest, y_model)
disp = ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
plt.show()
print("--------------------------------------------")



# TRAINING A RANDOM FOREST MODEL FOR COMPARISON
print("RANDOM FOREST CLASSIFIER")
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(df, y)
print("7-fold Cross Validation Score List: ")
scoreList = cross_val_score(model, df, y, cv=7)
print(scoreList)
print("Average: ", sum(scoreList) / len(scoreList))
model.fit(df, y)
y_model = model.predict(Xftest)
score = accuracy_score(yftest, y_model)
# calculating and printing the accuracy score of the model on test data that model has never seen before
print("Accuracy score on test data: ", score)
mat = confusion_matrix(yftest, y_model)
disp = ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
plt.show()
print("--------------------------------------------")

