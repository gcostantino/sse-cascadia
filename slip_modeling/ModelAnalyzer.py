import numpy as np

from FUNCTIONS.functions_slab import UTM_GEO
from utils.geometry_utils import load_cascadia_geometry
from utils.gnss_utils import load_cascadia_selected_stations
from utils.slip_model_utils import load_slip_models


class ModelAnalyzer:
    def __init__(self, preferred_lambda=100, preferred_sigma_m=.2, preferred_sigma_d=.1, shear_modulus=30e9):
        self.preferred_lambda = preferred_lambda  # km
        self.preferred_sigma_m = preferred_sigma_m  # mm
        self.preferred_sigma_d = preferred_sigma_d
        self.slip_model = load_slip_models(self.preferred_lambda, self.preferred_sigma_m, self.preferred_sigma_d)
        station_codes, station_coordinates = load_cascadia_selected_stations()
        geometry, green_td = load_cascadia_geometry()
        self.station_codes = station_codes
        self.station_coordinates = station_coordinates
        self.fault_geometry = geometry
        self.G = green_td
        self.strike_slip_rates = self.slip_model[:, :(len(geometry[:, 9]))]
        self.dip_slip_rates = self.slip_model[:, (len(geometry[:, 9])):]
        self.slip_rates = np.sqrt(self.dip_slip_rates ** 2 + self.strike_slip_rates ** 2)  # in mm
        self.signed_slip_rates = np.sign(self.dip_slip_rates) * self.slip_rates
        self.area = self.fault_geometry[:, 21] * 1e+6  # in m^2
        self.slip_potency_rate = self.area * self.slip_rates * 1e-03  # slip converted to meters
        self.signed_slip_potency_rate = self.area * self.signed_slip_rates * 1e-03  # slip converted to meters
        self.shear_modulus = shear_modulus  # GPa
        self.mo_rates = self.shear_modulus * self.slip_potency_rate
        self.signed_mo_rates = self.shear_modulus * self.signed_slip_potency_rate
        x_centr_lon, y_centr_lat = UTM_GEO(self.fault_geometry[:, 9], self.fault_geometry[:, 10])
        self.x_centr_lon = x_centr_lon
        self.y_centr_lat = y_centr_lat

    def analyze_model(self):
        # Code to analyze the inverted models
        pass


    def save_results(self, filepath):
        # Save analysis results to a file
        pass
