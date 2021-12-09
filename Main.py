import csv
import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# reading the data from the csv file
data = []
with open('Bank_salary.csv', 'r') as f:
    my_reader = csv.reader(f)
    for row in my_reader:
        data.append(row[0].split(';'))

# excluding the titles row
data = data[1:]

# replacing Male as 0 and Female as 1
for row in data:
    if row[-1] == 'Male':
        row[-1] = 0
    else:
        row[-1] = 1

# converting strings to floats
for row in data:
    for e in row:
        row[row.index(e)] = float(e)

# our data set is ready at this point

# writing data to csv file
with open('data_set.csv', 'w') as f:
    my_writer = csv.DictWriter(f, fieldnames=("Employee #", "Salary", "Years of Experience", "Gender"))
    my_writer.writeheader()
    for row in data:
        my_writer.writerow({"Employee #": row[0], "Salary": row[1], "Years of Experience": row[2], "Gender": row[3]})

# splitting data into y and X
data = np.array(data)
X = data[:, 2:]
y = data[:, 1:2]

# performing linear regression
betaHeadMatrix, standardErrors, confidanceIntervals = LinearRegression.linearRegression(X, y)

# plotting the results
Xfemale = np.array([x.tolist() for x in X if x[1] == 1])
yPredictions = np.matmul(np.c_[np.ones(len(Xfemale)), Xfemale], betaHeadMatrix)
yreal = [y[i].tolist() for i in range(len(X)) if X[i][1] == 1]
yreal.sort()
xaxis = Xfemale[:, 0:1].transpose()[0].tolist()
xaxis.sort()
yaxis = yPredictions.transpose()[0].tolist()
yaxis.sort()
plt.plot(xaxis, yaxis, label="Female")
plt.plot(xaxis, yreal, linestyle="", marker=".")

Xmale = np.array([x.tolist() for x in X if x[1] == 0])
yPredictions = np.matmul(np.c_[np.ones(len(Xmale)), Xmale], betaHeadMatrix)
yreal = [y[i].tolist() for i in range(len(X)) if X[i][1] == 0]
yreal.sort()
xaxis = Xmale[:, 0:1].transpose()[0].tolist()
xaxis.sort()
yaxis = yPredictions.transpose()[0].tolist()
yaxis.sort()
plt.plot(xaxis, yaxis, label="Male")
plt.plot(xaxis, yreal, linestyle="", marker=".")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()
