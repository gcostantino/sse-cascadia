import numpy as np
from skimage.measure import EllipseModel


class ConvergenceError(Exception):
    def __init__(self, x, message="Singular matrix: ellipse fitting did not succeed."):
        self.x = x
        self.message = f"{message}: {x}"
        super().__init__(self.message)


def fit_ellipse_mo(x, y, mo_rates, mo_thresh):
    """Fit a 2D ellipse to mo rates > mo_thresh."""
    em = EllipseModel()
    mask_ell = mo_rates > mo_thresh * np.max(mo_rates)
    x_thresh, y_thresh = x[mask_ell], y[mask_ell]
    try:
        em.estimate(np.vstack((x_thresh, y_thresh)).T)
    except TypeError as e:
        raise ConvergenceError(e)
    # xc, yc, a, b, theta = em.params
    if em.params is None:
        raise ConvergenceError(None)
    return em.params


def median_xy_mo(x, y, mo_rates, mo_thresh):
    """Warning: can return NaNs."""
    mask_ell = mo_rates > mo_thresh * np.max(mo_rates)
    x_thresh, y_thresh = x[mask_ell], y[mask_ell]
    return np.median(x_thresh), np.median(y_thresh)
