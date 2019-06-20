import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression



def get_base():
    tmp = pd.read_csv("C:/hello_world/Python/AI/lab_0/databases/online_shoppers_intention.csv", header=None)
    df = pd.DataFrame({"Administrative" : tmp[0][1:], "Administrative_Duration" : tmp[1][1:], "Informational" : tmp[2][1:],\
                       "Informational_Duration" : tmp[3][1:], "ProductRelated" : tmp[4][1:], "ProductRelated_Duration" : tmp[5][1:],\
                       "BounceRates" : tmp[6][1:], "ExitRates" : tmp[7][1:], "PageValues" : tmp[8][1:], "SpecialDay" : tmp[9][1:], "Month" : tmp[10][1:],\
                       "OperatingSystems" : tmp[11][1:], "Browser" : tmp[12][1:], "Region" : tmp[13][1:], "TrafficType" : tmp[14][1:], "VisitorType" : tmp[15][1:],\
                       "Weekend" : tmp[16][1:], "Revenue" : tmp[17][1:]})# now iteration from 1, not 0
    return df


    
df = get_base()
months = {"Feb" : 0., "Mar" : 1., "May" : 2., "June" : 3., "Jul" : 4., "Aug" : 5., "Sep" : 6., "Oct" : 7., "Nov" : 8., "Dec" : 9.}

for i in list(df.columns):
    for j in range(1, len(df[i]) + 1):
        if df[i][j] == "Returning_Visitor" or df[i][j] == "FALSE":
            df[i][j] = 0.
        elif df[i][j] == "New_Visitor" or df[i][j] == "TRUE":
            df[i][j] = 1.
        elif df[i][j] == "Other":
            df[i][j] = 2.
        elif df[i][j] in months.keys():
            df[i][j] = months[df[i][j]]
        else:
            df[i][j] = float(df[i][j])

np.random.seed(111)
df = np.random.permutation(df)

y = df.T[13].T
X = np.vstack((df.T[:13], df.T[14:])).T

y_train = y[:int(len(y) * 0.7)]
X_train = X[:int(len(X) * 0.7)]

print(y_train.shape)
print(y_train)
print(X_train.shape)
print(X_train)


y_test = y[int(len(y) * 0.7):]#     финальный тест на точность нашей модели (остальные 20% базы (которые НЕ встречались при тренировке))
X_test = X[int(len(X) * 0.7):]

LR = LogisticRegression()
LR.fit(X_train, y_train)

y_test_pred = LR.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis = 0) / X_test.shape[0]

print("Accuracy is: %.2f%%" % (acc * 100))


