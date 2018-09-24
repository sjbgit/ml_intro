import numpy as np

image = np.array([1,2,3,4,5]) 

result = image[:] < 3

print(result)