from denoised_data_utils.DenoisedDataLoader import DenoisedDataLoader


class DenoisedDataHandler:
    def __init__(self, window_length=60, offset=20):
        self.ddl = DenoisedDataLoader()
        self.window_length = window_length
        self.offset = offset
        self._corrected_time = None

    @property
    def corrected_time(self):
        if self._corrected_time is None:
            time_array = self.ddl.get_time_array()
            corrected_time = time_array[0 + self.offset:-self.window_length - self.offset]
            self.corrected_time = corrected_time
        return self._corrected_time

    @corrected_time.setter
    def corrected_time(self, value):
        self._corrected_time = value
