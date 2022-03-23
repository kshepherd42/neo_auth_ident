import pandas as pd
import numpy as np
data = pd.read_hdf('c_cpp_h_program_cc9995_vs4048_a0.5_l-1_train.h5')
sandhdf = pd.HDFStore('sandbox.h5')
sandhdf.put("stff", data[:30000])
