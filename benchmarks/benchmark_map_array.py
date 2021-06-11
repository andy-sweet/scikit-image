import numpy as np
from skimage.util._map_array import map_array, map_array_dense, map_array_flat


low = 0
high = 127


class MapArrayBenchmarks:

    param_dict = {
            'size': (256, 512, 1024, 2048),
            'dtype': (np.int8, np.int16, np.int32, np.int64),
    }
    param_names = param_dict.keys()
    params = param_dict.values()

    def setup(self, size, dtype):
        rng = np.random.default_rng(0)
        self.in_arr = rng.integers(low=low, high=high, size=(size, size), dtype=dtype)
        self.in_values = np.unique(self.in_arr)
        self.out_values = rng.integers(low=low, high=high, size=self.in_values.shape, dtype=dtype)

    def time_ndarray(self, size, dtype):
        self.out_values[self.in_arr]

    def time_map_array(self, size, dtype):
        map_array(self.in_arr, self.in_values, self.out_values)

    def time_map_array_dense(self, size, dtype):
        map_array_dense(self.in_arr, self.in_values, self.out_values)

    def time_map_array_flat(self, size, dtype):
        map_array_flat(self.in_arr, self.in_values, self.out_values)

