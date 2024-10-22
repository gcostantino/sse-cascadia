import numpy as np


def custom_fft(x, sf, n_points, axis=0, pos_freq_only=True):
    spectrum = np.fft.fft(x, axis=axis)
    freq = np.fft.fftfreq(n_points, d=1 / sf)
    spectrum_shifted = np.fft.fftshift(spectrum)
    freq_shifted = np.fft.fftshift(freq)
    if pos_freq_only:
        positive_freq_mask = freq_shifted > 0
        positive_freq = freq_shifted[positive_freq_mask]
        positive_freq_idx = np.where(positive_freq_mask)[0]
        # positive_spectrum = np.abs(spectrum_shifted)[positive_freq_mask]
        # positive_spectrum = np.abs(spectrum_shifted)[:, positive_freq_mask]
        positive_spectrum = np.take(np.abs(spectrum_shifted), positive_freq_idx, axis=axis)
        '''index_tuple = [slice(None)] * spectrum_shifted.ndim
        index_tuple[axis] = positive_freq_mask
        result = np.abs(spectrum_shifted)[tuple(index_tuple)]'''
        spectrum_shifted = positive_spectrum
        freq = positive_freq  # / 365
    return freq, spectrum_shifted
