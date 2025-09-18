import os
import numpy as np
import utm
from matplotlib import pyplot as plt
from pyproj import Transformer
from shapely.ops import unary_union

from data_config_files import v7_parameters as params
from sse_generator_v2.synthetic_sse_cascadia import synthetic_displacements_stations_cascadia_v4
from utility_methods import cascadia_coordinates
from shapely.geometry import Polygon

np.random.seed(42)


def modified_generate_synthetic_sses(n_samples, station_coordinates, **kwargs):
    """As 'synthetic_sses_v3' but:
    - when a dislocation is computed, this will be drawn in a vicinity of existing dislocations with probability p,
    otherwise its position will be randomly generated with probability 1-p."""
    synthetic_displacement, catalogue = synthetic_displacements_stations_cascadia_v4(n_samples, station_coordinates,
                                                                                     **kwargs)
    return synthetic_displacement, catalogue


def modified_synthetic_time_series(n_samples, station_coordinates, **kwargs):
    """Similar to synthetic_time_series_real_gaps_extended_v6(). What's new:
    - 30 dislocations are built, with simple spatial correlation. For details see synthetic_sses_v4 and reference
    therein."""
    max_n_disloc = kwargs.pop('max_n_disloc', 30)
    aspect_ratio = kwargs.pop('aspect_ratio', 1)
    correct_latlon = kwargs.pop('correct_latlon', True)
    spatial_proba = kwargs.pop('spatial_proba', 0.7)
    min_centroid_distance = kwargs.pop('min_centroid_distance', 0)
    max_centroid_distance = kwargs.pop('max_centroid_distance', 50)

    synthetic_displacement, catalogue = modified_generate_synthetic_sses(n_samples, station_coordinates,
                                                                         max_n_disloc=max_n_disloc,
                                                                         aspect_ratio=aspect_ratio,
                                                                         correct_latlon=correct_latlon,
                                                                         spatial_proba=spatial_proba,
                                                                         min_centroid_distance=min_centroid_distance,
                                                                         max_centroid_distance=max_centroid_distance,
                                                                         **kwargs)

    return catalogue, synthetic_displacement


def square_vertices_3d(center, strike_deg, dip_deg, L, L_dip=None):
    """
    center: (x, y, z)  (x east, y north, z up)
    strike_deg: degrees clockwise from north
    dip_deg: degrees positive downward from horizontal
    L: side length along strike (meters)
    L_dip: side length along dip (meters). If None, use L (square).
    returns: list of 4 vertices as (x,y,z) in counter-clockwise order (then closed if desired)
    """
    x0, y0, z0 = center
    if L_dip is None:
        L_dip = L
    # convert to radians
    theta = np.deg2rad(strike_deg)
    delta = np.deg2rad(dip_deg)

    # unit vector along strike (horizontal)
    u_s = np.array([np.sin(theta), np.cos(theta), 0.0])

    # unit vector down-dip (in-plane, pointing downwards)
    # horizontal azimuth = strike + 90 deg -> sin/cos identities give below compact form
    u_d = np.array([np.cos(delta) * np.cos(theta),
                    -np.cos(delta) * np.sin(theta),
                    -np.sin(delta)])
    # normalize (should already be unit, but numerically safe)
    u_s = u_s / np.linalg.norm(u_s)
    u_d = u_d / np.linalg.norm(u_d)

    hs = L / 2.0
    hd = L_dip / 2.0

    # four corners: center +/- hs*u_s +/- hd*u_d
    corners = []
    for sx in [+1, -1]:
        for sd in [+1, -1]:
            p = np.array([x0, y0, z0]) + sx * hs * u_s + sd * hd * u_d
            corners.append(tuple(p))
    # ordering above gives: (+,+), (+,-), (-,+), (-,-) — you can reorder to ccw if needed
    return corners


def get_coordinates_of_reprojected_dislocation(x_center, y_center, L, W, strike, dip):
    alpha = np.pi / 2 - strike
    x_fault = L / 2 * np.cos(alpha) * np.array([-1., 1., 1., -1.]) + np.sin(alpha) * np.cos(dip) * W / 2 * np.array(
        [-1., -1., 1., 1.])
    y_fault = L / 2 * np.sin(alpha) * np.array([-1., 1., 1., -1.]) + np.cos(alpha) * np.cos(dip) * W / 2 * np.array(
        [1., 1., -1., -1.])
    x_fault += x_center
    y_fault += y_center
    return x_fault, y_fault


def square_from_center(xc, yc, L):
    """Return a shapely Polygon of a square centered at (xc, yc) with side L."""
    half = L / 2.0
    coords = [
        (xc - half, yc - half),
        (xc + half, yc - half),
        (xc + half, yc + half),
        (xc - half, yc + half)
    ]
    return coords


if __name__ == '__main__':
    n_samples = 5000

    n_stations = params.n_stations
    window_length = params.window_length
    depth_range = params.depth_range
    rake_range = params.rake_range
    data_gap_proba = params.data_gap_proba
    slip_range = params.slip_range
    max_n_disloc = params.max_n_disloc
    spatial_proba = params.spatial_proba
    min_centroid_distance = params.min_centroid_distance
    max_centroid_distance = params.max_centroid_distance
    noise_has_trend = params.noise_has_trend

    codes, coords = cascadia_coordinates()
    dset_type = 'denois'
    base_dir = os.path.expandvars('$WORK')  # '.'

    dataset_filename = f'{dset_type}_synth_ts_cascadia_realgaps_extended_v7_{n_stations}stations_depth_{depth_range[0]}_{depth_range[1]}_20x20_{max_n_disloc}_disloc_slip_{slip_range[0]}_{slip_range[1]}_noise_with_trend.data'

    dataset_path = os.path.join(base_dir, dataset_filename)

    catalogue, _ = modified_synthetic_time_series(n_samples, coords, slip_range=slip_range, max_n_disloc=max_n_disloc,
                                                  spatial_proba=spatial_proba,
                                                  min_centroid_distance=min_centroid_distance,
                                                  max_centroid_distance=max_centroid_distance, )

    lat0, lon0 = 48, -125
    zone_num, zone_letter = utm.from_latlon(lat0, lon0)[2:4]
    epsg = 32600 + zone_num if zone_letter >= 'N' else 32700 + zone_num

    # transformer: WGS84 -> UTM (meters)
    transformer = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)

    # transform origin and points
    x0_m, y0_m = transformer.transform(lon0, lat0)

    L = 20  # km
    shear_modulus = 30e9
    stress_drop_list = []
    for n in range(n_samples):
        x_array, y_array, z_array = np.array(catalogue[n][0]), np.array(catalogue[n][1]), np.array(catalogue[n][2])
        strike_array, dip_array = np.array(catalogue[n][3]), np.array(catalogue[n][4])
        slip_array = np.array(catalogue[n][6]) * 1e-3  # meters
        disloc_polygons, tot_slip = [], []
        for j in range(len(x_array)):  # n disloc
            lat, lon, strike, dip, slip = x_array[j], y_array[j], strike_array[j], dip_array[j], slip_array[j]
            x_m, y_m = transformer.transform(lon, lat)
            x_km, y_km = (x_m - x0_m) / 1000.0, (y_m - y0_m) / 1000.0  # shift relative to origin, convert to km
            square_coords = square_from_center(x_km, y_km, L)  # square_vertices_3d((x_km, y_km, z), strike, dip, L, L_dip=None)
            # square_coords = get_coordinates_of_reprojected_dislocation(x_km, y_km, L, L, strike, dip)
            poly = Polygon([t[:2] for t in square_coords])
            if poly.is_valid:
                disloc_polygons.append(poly)
                tot_slip.append(slip)
            else:
                print('polygon not valid')  # diagnose
        total_slip_area = unary_union(disloc_polygons).area * 1e3 # m^2
        radius = np.sqrt(total_slip_area / np.pi)  # circular crack approx.
        print('total_slip_area', total_slip_area, 'radius', radius)
        stress_drop_event = shear_modulus * np.mean(slip_array) / radius
        print('stress_drop_event', stress_drop_event * 1e-3, 'kPa')
        stress_drop_list.append(stress_drop_event)

    stress_drop_list = np.array(stress_drop_list)
    plt.hist(stress_drop_list * 1e-6, bins=20)
    plt.axvline(0.17, color='r')
    plt.xlabel('Stress drop (MPa)')
    plt.ylabel('Count')
    plt.show()
    exit(0)
    # epi_lat_all, epi_lon_all, hypo_depth_all, strike_all, dip_all, rake_all, u_all = [], [], [], [], [], [], [], []
