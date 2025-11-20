import tphate
import numpy as np

data = np.random.rand(500,200)
tphate_op = tphate.TPHATE()
data_tphate = tphate_op.fit_transform(data)
