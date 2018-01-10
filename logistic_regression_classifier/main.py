import numpy as np 
from sklearn import linear_model 
import matplotlib.pyplot as plt
from sklearn import metrics, cross_validation

# Define sample input data
X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]]) 
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=42)

# Create the logistic regression classifier 
classifier = linear_model.LogisticRegression(solver='liblinear', C=100)

# Train the classifier 
classifier.fit(x_train, y_train)

# Run the classifier on the mesh grid 
output = classifier.predict(x_test)

score = metrics.accuracy_score(y_test, classifier.predict(x_test))

print('Accuracy: {0:f}'.format(score))
