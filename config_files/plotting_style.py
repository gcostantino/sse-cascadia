import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 16
LARGE_SIZE = 20
FONT_FAMILY = 'Helvetica Neue'

style_dict = {
    'font.size': SMALL_SIZE,
    'font.family': FONT_FAMILY,
    'axes.titlesize': SMALL_SIZE,
    'axes.labelsize': MEDIUM_SIZE,
    'xtick.labelsize': SMALL_SIZE,
    'ytick.labelsize': SMALL_SIZE,
    'legend.fontsize': SMALL_SIZE,
    'figure.titlesize': LARGE_SIZE
}


def set_matplotlib_style():
    plt.rcParams.update(style_dict)


def get_style_dict():
    # return plt.rcParams
    return style_dict


def get_style_attr(attr):
    return plt.rcParams[attr]


def temporary_matplotlib_syle(style_dict):
    return plt.rc_context(style_dict)
