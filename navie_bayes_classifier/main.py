import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from utilities import visualize_classifier

# Input file containing
input_file = 'data_multivar_nb.txt'

# Load data from input file 
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=3)

# Create Naïve Bayes classifier 
classifier = GaussianNB()

# Train the classifier
classifier.fit(X_train,y_train)

y_test_pred = classifier.predict(X_test)

# compute accuracy of the classifier
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]

# compute accuracy of the classifier 
print("Accuracy of the new classifier =", round(accuracy, 2), "%")

# Visualize the performance of the classifier
visualize_classifier(classifier, X_test, y_test)

num_folds = 3 
accuracy_values = cross_validation.cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds) 
print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%")

precision_values = cross_validation.cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds) 
print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%")

recall_values = cross_validation.cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%")

f1_values = cross_validation.cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")