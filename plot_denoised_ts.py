import joblib
import numpy as np

from FUNCTIONS.functions_figures import overview_latitude_time_plot
from config_files.plotting_style import set_matplotlib_style
from utils.catalogue_parsing import tremor_catalogue

set_matplotlib_style()

if __name__ == '__main__':
    preferred_L = 100  # km
    preferred_sigma_m = 0.2  # mm
    preferred_sigma_d = 0.1
    slip_model = joblib.load(
        f'../../DATA/sse-cascadia/inversed_ts/dump_50km/slip_inversion_sigma_d_01/inversed_ts_slip5_1000_noise_with_trend_demean_corrlength_{preferred_L}km_sigmam_{preferred_sigma_m}_sigmanoise3D_{preferred_sigma_d}')
    geometry, recs = joblib.load('../../DATA/sse-cascadia/cascadia_geometry_info/geometry_cascadia')
    green_td = joblib.load('../../DATA/sse-cascadia/cascadia_geometry_info/green_td_cascadia')
    station_codes = np.loadtxt('INPUT_FILES/stations_cascadia_200.txt', usecols=0, dtype=np.str_).tolist()
    station_coordinates = np.loadtxt('INPUT_FILES/stations_cascadia_200.txt', usecols=(1, 2))

    with np.load('../../DATA/sse-cascadia/denoised_ts/denoised_ts_slip5_1000_noise_with_trend_demean.npz') as f:
        denoised_ts, time_vec = f['data'], f['time']
    import matplotlib.pyplot as plt

    print(station_codes[0])
    print('tot EW displacement', np.sum(denoised_ts[:,0,0]), '-->', np.sum(denoised_ts[:,0,0]) / len(time_vec) * 365, 'mm/yr')

    '''plt.plot(denoised_ts[:,0,0])
    plt.show()'''

    window_length = 60
    offset = 20
    # corrected_time = time_vec[0 + offset:-window_length - offset]

    tremors = tremor_catalogue()
    tremors = tremors[(tremors[:, 0] > 40) & (tremors[:, 0] < 50)]
    tremors = tremors[(tremors[:, 3] > 2007) & (tremors[:, 3] < 2023 - (window_length + 2 * offset) / 365)]
    latsort = np.argsort(station_coordinates[:, 0])[::-1]

    overview_latitude_time_plot(time_vec, denoised_ts, tremors, station_coordinates, latsort, offset=20,
                                window_length=60,
                                static=False, downsample_tremors=True, draw_tremors=True, tremor_alpha=1.,
                                data_pcolormesh=False, zoom=False, show=False, dpi=300, trenchwards_only=True,
                                save_as='png', raster_scatter=False, modified_cmap=True)
