import numpy as np

fQ_arr_with_exceptions = np.arange(-50, 51, 1)
index = np.argwhere(fQ_values == -10)
fQ_array = np.delete(fQ_arr_with_exceptions, index)/10.0
