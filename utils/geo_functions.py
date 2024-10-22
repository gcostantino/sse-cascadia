import numpy as np


def logmo_to_mw(logmo):
    return (logmo - 9.1) / 1.5


def mo_to_mw(mo):
    return (np.log10(mo) - 9.1) / 1.5


def mw_to_logmo(mw):
    return 1.5 * mw + 9.1


def mw_to_mo(mw):
    return 10 ** (1.5 * mw + 9.1)
