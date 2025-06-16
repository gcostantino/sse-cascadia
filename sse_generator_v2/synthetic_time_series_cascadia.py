import numpy as np

from sse_generator_v2.artificial_noise_cascadia import generate_artificial_noise
from sse_generator_v2.synthetic_sse_cascadia import generate_synthetic_sses


def synthetic_time_series(n_samples, n_selected_stations, window_length=60, **kwargs):
    """Similar to synthetic_time_series_real_gaps_extended_v6(). What's new:
    - 30 dislocations are built, with simple spatial correlation. For details see synthetic_sses_v4 and reference
    therein."""
    reference_period = (2007, 2023)
    max_n_disloc = kwargs.pop('max_n_disloc', 30)
    aspect_ratio = kwargs.pop('aspect_ratio', 1)
    correct_latlon = kwargs.pop('correct_latlon', True)
    spatial_proba = kwargs.pop('spatial_proba', 0.7)
    min_centroid_distance = kwargs.pop('min_centroid_distance', 0)
    max_centroid_distance = kwargs.pop('max_centroid_distance', 50)
    noise_has_trend = kwargs.pop('noise_has_trend', True)
    noise_windows, station_codes, station_coordinates = generate_artificial_noise(n_samples,
                                                                                            n_selected_stations,
                                                                                            window_length,
                                                                                            reference_period,
                                                                                            p=kwargs['p'],
                                                                                            add_trend=noise_has_trend)

    transients, random_durations, synthetic_displacement, catalogue = generate_synthetic_sses(n_samples, window_length,
                                                                                        station_codes,
                                                                                        station_coordinates,
                                                                                        max_n_disloc=max_n_disloc,
                                                                                        aspect_ratio=aspect_ratio,
                                                                                        correct_latlon=correct_latlon,
                                                                                        spatial_proba=spatial_proba,
                                                                                        min_centroid_distance=min_centroid_distance,
                                                                                        max_centroid_distance=max_centroid_distance,
                                                                                        **kwargs)

    synthetic_data = noise_windows + transients
    synthetic_data = np.nan_to_num(synthetic_data, nan=0.)  # NaNs are put back to zero

    rnd_idx = np.random.permutation(len(synthetic_displacement))  # not really necessary, but we do it anyways
    synthetic_data = synthetic_data[rnd_idx]
    random_durations = random_durations[rnd_idx]
    transients = transients[rnd_idx]
    catalogue = [catalogue[i] for i in rnd_idx]  # catalogue[rnd_idx]
    synthetic_displacement = [synthetic_displacement[i] for i in rnd_idx]  # synthetic_displacement[rnd_idx]
    return synthetic_data, random_durations, catalogue, synthetic_displacement, transients, station_codes, station_coordinates
