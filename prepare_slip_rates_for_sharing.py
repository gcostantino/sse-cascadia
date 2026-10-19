"""
Author: Giuseppe Costantino
27 July 2026
"""
import joblib
import numpy as np
from pyproj import Transformer


""""""""" PROJECTIONS """""""""
def GEO_UTM(longitude, latitude):
    # CONVERSION THE GEOGRAFICAL COORDINATES EPSG:4326 TO MERCANTOUR COORDINATES EPSG:3857
    longitude, latitude = np.array(longitude), np.array(latitude)
    TRAN_4326_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    (x, y) = TRAN_4326_TO_3857.transform(longitude, latitude)

    return x / 1.E3, y / 1.E3


def UTM_GEO(x, y):
    # CONVERSION THE MERCANTOUR COORDINATES EPSG:3857 TO GEOGRAFICAL COORDINATES EPSG:4326
    x, y = np.array(x), np.array(y)  # CONVERT BACK TO METERS
    TRAN_3857_TO_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)  # DEFINE THE TRANSFORMATION

    return TRAN_3857_TO_4326.transform(x * 1.E3, y * 1.E3)  # RETURN THE COORDIANTES IN DEGREES


if __name__ == '__main__':
    shear_modulus = 30e9  #use it to compute magnitudes

    geometry, _ = joblib.load('sharing/geometry_cascadia')
    green_functions = joblib.load('sharing/green_td_cascadia')
    slip_model = joblib.load('sharing/slip_rates')
    time_array = joblib.load('sharing/time_array')

    # Information on the (triangular) mesh format
    # rectangle_x, rectangle_y, rectangle_z, width_km, height_km, area_ratio_km2, LonLatD_to_XYZ, Dip, Strike, Centroid_X, Centroid_Y, Centroid_Z, Vertex1_X, Vertex1_Y, Vertex1_Z,\ Vertex2_X, Vertex2_Y, Vertex2_Z, Vertex3_X, Vertex3_Y, Vertex3_Z, Area
    # You will probably need the following elements : Dip, Strike, Centroid_X, Centroid_Y, Centroid_Z, Area.
    #
    # slip_rates --> is a matrix containing time x slip rates, where the slip rates are a concatenation of strike slip and dip slip (twice the size of the geometry). Slip is in mm.

    n_triangles = len(geometry[:, 9])
    n_time_steps = len(time_array)

    print(f'Mesh contains {n_triangles} triangles')
    print('Number of time steps:', n_time_steps)

    print(f'Analyzed period: {time_array[0]} - {time_array[-1]}')

    dip_slip_component = slip_model[:, :n_triangles]
    strike_slip_component = slip_model[:, n_triangles:]


    slip_rates = np.sqrt(dip_slip_component ** 2 + strike_slip_component ** 2)

    print('Shape of slip rates:', slip_rates.shape, f'---> ({n_time_steps} time_steps x {n_triangles} triangles)')

    # you might want to convert fault centroids to/from lat/lon and UTM
    x_centr_lon, y_centr_lat = UTM_GEO(geometry[:, 9], geometry[:, 10])
    # similarly, you can use the GEO_UTM

    # to load the station codes and coordinates used in this study, please do:
    station_codes = np.loadtxt('sharing/stations_cascadia_200.txt', usecols=0, dtype=np.str_).tolist()
    station_coordinates = np.loadtxt('sharing/stations_cascadia_200.txt', usecols=(1, 2))

    print('Number of stations:', len(station_codes))

    # you might want to manipulate the Green's functions
    print(green_functions.shape)

    # this tensor has shape (n_triangles, n_stations, n_gps_components, n_slip_components) --> (1358, 200, 3, 3)
    # n_triangles = 1358
    # number of stations: 200
    # number of GPS components: 3, where
    #  - idx 0: E-W
    #  - idx 1: N-S
    #  - idx 2: vertical (not used here)
    # number of slip components: 3, where
    #  - idx 0: dip-slip
    #  - idx 1: strike-slip
    #  - idx 2: tensile (not used here)

    # slip_rates contains the full slip history along the Cascadia margin.
    # To extract the slow slip events, you can refer to the catalogue dates, but
    # for a precise extraction and to reproduce all the results of the paper, refer to the original code and use
    # the class SlowSlipEventExtractor(...).
    # the code is free to use and available on GitHub at: https://github.com/gcostantino/sse-cascadia
    # Also, feel free to contact me at giuseppe.costantino@ens.fr
