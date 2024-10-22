import joblib


def load_slip_models(preferred_L=100, preferred_sigma_m=.2, preferred_sigma_d=.1):
    slip_model = joblib.load(
        f'../../DATA/sse-cascadia/inversed_ts/dump_50km/slip_inversion_sigma_d_01/inversed_ts_slip5_1000_noise_with_trend_demean_corrlength_{preferred_L}km_sigmam_{preferred_sigma_m}_sigmanoise3D_{preferred_sigma_d}')
    return slip_model
