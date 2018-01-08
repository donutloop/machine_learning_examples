
# We use the process of normalization to modify the values in the feature vector so that we can measure them on a common scale. 
# In machine learning, we use many different forms of normalization. 
# Some of the most common forms of normalization aim to modify the values so that they sum up to 1 . L1 normalization ,
# which refers to Least A bsolute Deviations , works by making sure that the sum of absolute values is 1 in each row. L2 normalization ,
# which refers to least squares, works by making sure that the sum of squares is 1 .
# In general, L1 normalization technique is considered more robust than L2 normalization technique.
# L1 normalization technique is robust because it is resistant to outliers in the data.
# A lot of times, data tends to contain outliers and we cannot do anything about it.
# We want to use techniques that can safely and effectively ignore them during the calculations.
# If we are solving a problem where outliers are important, then maybe L2 normalization becomes a better choice.

import numpy as np 
from sklearn import preprocessing

input_data = np.array(
    [[5.1, -2.9, 3.3],
    [-1.2, 7.8, -6.1],
    [3.9, 0.4, 2.1],
    [7.3, -9.9, -4.5]]
)

# Normalize data 
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1') 
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nL1 normalized data:\n", data_normalized_l1) 
print("\nL2 normalized data:\n", data_normalized_l2)