import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 16
LARGE_SIZE = 20
FONT_FAMILY = 'Helvetica Neue'

def set_matplotlib_style():
    plt.rcParams.update({
        'font.size': SMALL_SIZE,
        'font.family': FONT_FAMILY,
        'axes.titlesize': SMALL_SIZE,
        'axes.labelsize': MEDIUM_SIZE,
        'xtick.labelsize': SMALL_SIZE,
        'ytick.labelsize': SMALL_SIZE,
        'legend.fontsize': SMALL_SIZE,
        'figure.titlesize': LARGE_SIZE
    })
