import numpy as np


class DenoisedDataLoader:
    def __init__(self,
                 filepath='../../DATA/sse-cascadia/denoised_ts/denoised_ts_slip5_1000_noise_with_trend_demean.npz'):
        self.filepath = filepath
        self.denoised_ts = None
        self.time_array = None

    def _load_denoised_data(self):
        with np.load(self.filepath) as f:
            denoised_ts, time_array = f['data'], f['time']
        self.denoised_ts = denoised_ts
        self.time_array = time_array

    def get_denoised_data(self):
        if self.denoised_ts is None and self.time_array is None:
            self._load_denoised_data()
        return self.time_array, self.denoised_ts
