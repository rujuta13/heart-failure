# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:/heart_failure_train.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[2,3,7,9,11])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

def KNN():
    # Training the KNN model on the Training set
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    classifier.fit(x_train, y_train)

    # Predicting the results
    x1 = pd.read_csv('D:/heart_failure_test.csv').values
    x1 = np.array(ct.fit_transform(x1))
    x1 = sc.transform(x1)
    y_pred = classifier.predict(x1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    return y_pred

def Log_Reg():
    # Training the Logistic Regression model on the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)

    # Predicting the results
    x1 = pd.read_csv('D:/heart_failure_test.csv').values
    x1 = np.array(ct.fit_transform(x1))
    x1 = sc.transform(x1)
    y_pred = classifier.predict(x1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    return y_pred

def ran_forest():
    # Training the Random Forest model on the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
    classifier.fit(x_train, y_train)

    # Predicting the results
    x1 = pd.read_csv('D:/heart_failure_test.csv').values
    x1 = np.array(ct.fit_transform(x1))
    x1 = sc.transform(x1)
    y_pred = classifier.predict(x1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    return y_pred

def dec_tree():
    # Training the Decision Tree model on the Training set
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='entropy')
    classifier.fit(x_train, y_train)

    # Predicting the results
    x1 = pd.read_csv('D:/heart_failure_test.csv').values
    x1 = np.array(ct.fit_transform(x1))
    x1 = sc.transform(x1)
    y_pred = classifier.predict(x1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    return y_pred

def ksvm():
    # Training the Kernel SVM model on the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf')
    classifier.fit(x_train, y_train)

    # Predicting the results
    x1 = pd.read_csv('D:/heart_failure_test.csv').values
    x1 = np.array(ct.fit_transform(x1))
    x1 = sc.transform(x1)
    y_pred = classifier.predict(x1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    return y_pred

#Storing data in csv file

col1 = KNN().flatten()
col2 = Log_Reg().flatten()
col3 = ran_forest().flatten()
col4 = dec_tree().flatten()
col5 = ksvm().flatten()

df=pd.DataFrame({'KNN':col1, 'Logistic Regression':col2, 'Random Forest':col3, 'Decision Tree':col4, 'Kernel SVM':col5}, index = np.arange(1,219))
df.to_csv('D:/ Heart Failure Predictions.csv')