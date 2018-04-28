import sklearn.datasets as d
import numpy as np
cls_set = d.make_classification(weights=[0.1])
np.bincount(cls_set[1])
print(cls_set)
