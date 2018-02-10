import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd 

# Look on kaggle for that data set
df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -9999, inplace=True)

# doesn't infulence if a human has cancer or not 
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop['class'], 1)
y = np.array(df['class'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()

clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)

print(accuracy)

example_measures = np.array([4,2,1,1,1,2,3,2,1])

prediction = clf.predict(example_measures)

print(prediction)