import pandas as pd
import numpy as np
data = pd.read_hdf('cpp_train.h5')
sandhdf = pd.HDFStore('sandbox.h5')
sandhdf.put("stff", data[:30000])
