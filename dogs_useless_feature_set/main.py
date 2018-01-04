import numpy as np 
import matplotlib.pyplot as plt 

greyhounds = 500
labs = 500

grey_heights = 28 + 4 * np.random.randn(greyhounds)
lab_heights = 24 + 4 * np.random.randn(labs)

plt.hist([grey_heights, lab_heights], stacked=True, color=['r', 'b'])
plt.show()