import pandas as pd 
import quandl 
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High','Adj. Low','Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High']-df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_Change'] = (df['Adj. Close']-df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-9999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
X = preprocessing.scale(X)
df.dropna(inplace=True)
y = np.array(df['label'])

# test size is set to 20 % of data 
x_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# n_jobs=-1 runs as many jobs as possible
clf = LinearRegression(n_jobs=-1)

# train on train data 
clf.fit(x_train, y_train)

# predict prices of test data
accuracy = clf.score(X_test, y_test)

print(accuracy)