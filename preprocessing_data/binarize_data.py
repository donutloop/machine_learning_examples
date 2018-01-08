import numpy as np 
from sklearn import preprocessing

input_data = np.array(
    [[5.1, -2.9, 3.3],
    [-1.2, 7.8, -6.1],
    [3.9, 0.4, 2.1],
    [7.3, -9.9, -4.5]]
)

# Binarization 
# This process is used when we want to convert our numerical values into boolean values.
# Let's use an inbuilt method to binarize input data using 2.1 as the threshold value.

# Binarize data 
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)

print("\nBinarized data:\n", data_binarized)