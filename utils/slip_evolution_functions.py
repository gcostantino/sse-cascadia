import numpy as np
from scipy.optimize import curve_fit


def sinusoidal_slip_evolution(t, A, T0, T1, T2):
    """From https://archuleta.faculty.geol.ucsb.edu/library/pdf/2006-PL-SC-RJA-BSSA.pdf"""

    evolution = np.zeros(t.shape)

    # First segment: T0 <= t <= T0 + T1
    mask1 = (t >= T0) & (t <= T0 + T1)
    evolution[mask1] = A * np.sin(np.pi * (t[mask1] - T0) / (2 * T1))

    # Second segment: T0 + T1 < t <= T0 + T1 + T2
    mask2 = (t > T0 + T1) & (t <= T0 + T1 + T2)
    evolution[mask2] = A / 2 * (1 + np.cos((t[mask2] - (T0 + T1)) * np.pi / T2))

    evolution[t < T0] = 0  # Setting values before T0 to 0
    return evolution


def triangular_slip_evolution(t, A, T0, T1, T2):
    """
    Creates a piecewise triangular function.
    """
    y = np.zeros(t.shape)

    rise_mask = (t >= T0) & (t <= T1)
    y[rise_mask] = A * (t[rise_mask] - T0) / (T1 - T0)

    fall_mask = (t > T1) & (t <= T2)
    y[fall_mask] = A * (T2 - t[fall_mask]) / (T2 - T1)

    return y


def fit_sin_cos_mo_evolution(y, t, window_length):
    def _model_slip_evolution(t, A, T0, T1, T2):
        y_model = sinusoidal_slip_evolution(t, A, T0, T1, T2)
        return y_model
    total_length, mo_rate_peak_time, max_mo_rate = len(y), np.argmax(y), np.max(y)
    tot_mo = np.sum(y)
    initial_guess = (tot_mo, 0, mo_rate_peak_time, window_length)
    bounds = ([0., -np.inf, 0, 0], [tot_mo, np.inf, window_length, window_length])
    opt_params, _ = curve_fit(_model_slip_evolution, t, y, p0=initial_guess, bounds=bounds)
    #A_fit, T0_fit, T1_fit, T2_fit = opt_params
    return opt_params


def fit_slip_evolution(t, y, slip_ev_fcn_signature=sinusoidal_slip_evolution, initial_guess=None, bounds=None,
                       var_on_peak=3):
    def _model_slip_evolution(*params):
        y_model = slip_ev_fcn_signature(*params)
        return y_model

    if initial_guess is None:
        total_length, slip_rate_peak_time, tot_slip = len(y), np.argmax(y), np.sum(y)
        initial_guess = (tot_slip, 0, slip_rate_peak_time, total_length)  # (works for both sinusoidal and triangular)
        #bounds = ([0., -np.inf, 0, 0], [tot_slip, np.inf, total_length, total_length])
        bounds = ([0., -np.inf, slip_rate_peak_time - var_on_peak, 0],
                  [tot_slip, np.inf, slip_rate_peak_time + var_on_peak, np.inf])
    opt_params, _ = curve_fit(_model_slip_evolution, t, y, p0=initial_guess, bounds=bounds)
    return opt_params

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t = np.linspace(0, 10, 100)
    y = triangular_slip_evolution(t, 2, 3, 6, 9)
    plt.scatter(t, y)
    plt.show()
