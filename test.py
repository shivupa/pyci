from utils import *
import numpy as np
hfdet = (frozenset([0,1,2,3,4]),frozenset([0,1,2,3,4]))
print(gen_dets_sets_truncated(7,5,5,[hfdet]))
print (np.shape(gen_dets_sets_truncated(7,5,5,[hfdet])))
