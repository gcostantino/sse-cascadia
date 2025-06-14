import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties

from FUNCTIONS.functions_figures import _annotate_line
from config_files.plotting_style import set_matplotlib_style
from sse_extraction.SlowSlipEventExtractor import SlowSlipEventExtractor
from utils.geo_functions import logmo_to_mw, mw_to_logmo, mw_to_mo, mo_to_mw
from utils.math_utils import straight_line


def _mo_duration_scaling(axis: Axes, se: SlowSlipEventExtractor, refine_durations: bool, legend: bool):
    mwax = axis.secondary_xaxis('top', functions=(logmo_to_mw, mw_to_logmo))

    mo_dummy_array = np.array([10, 25])

    # inset Axes
    x1, x2, y1, y2 = 14.5, 20, 4.8, 7  # subregion of the original plot
    inset_width = 0.57
    ax_inset = axis.inset_axes([0.4, 0.04, inset_width, inset_width], xlim=(x1, x2), ylim=(y1, y2), xticklabels=[],
                               yticklabels=[])
    secax_inset = ax_inset.secondary_xaxis('top', functions=(logmo_to_mw, mw_to_logmo))
    secax_inset.set_xticklabels([])

    for axx in (axis, ax_inset):
        axx.fill_between(mo_dummy_array, straight_line(mo_dummy_array, 1, -12), straight_line(mo_dummy_array, 1, -13),
                         color='C0', alpha=0.5)
        axx.fill_between(mo_dummy_array, straight_line(mo_dummy_array, 1 / 3, -15 / 3),
                         straight_line(mo_dummy_array, 1 / 3, -17 / 3),
                         color='C3', alpha=0.5)
        # add the 10^13 upper bound on Mo rates
        axx.plot(mo_dummy_array, straight_line(mo_dummy_array, 1, -13),
                         color='k', alpha=1., linestyle='--', lw=2.)

    for slip_thresh in se.slip_thresholds:
        event_moment_rate_list = se.get_moment_rate_events(slip_thresh, refine_durations)
        event_moment_list = np.array([mw_to_mo(mw) for mw in se.get_magnitude_events(slip_thresh, refine_durations)])
        event_duration_list = np.array([len(mo_rate) for mo_rate in event_moment_rate_list])

        valid_mo_idx = np.where(event_moment_list > 0.)[0]
        event_moment_list = event_moment_list[valid_mo_idx]
        event_duration_list = event_duration_list[valid_mo_idx]
        for axx in (axis, ax_inset):
            axx.scatter(np.log10(event_moment_list), np.log10(event_duration_list * 86400),
                        edgecolors=matplotlib.colors.colorConverter.to_rgba('black', alpha=.5),
                        label=f'thresh: {slip_thresh} mm/day')

    day_to_seconds = 86400.
    y_label_offset = -0.4
    x_label_pos = 11.3
    for axx in (axis, ax_inset):
        axx.axhline(y=np.log10(1 * day_to_seconds), color='grey', linestyle='--', lw=2., zorder=-2)
        axx.axhline(y=np.log10(30 * day_to_seconds), color='grey', linestyle='--', lw=2., zorder=-2)
        axx.axhline(y=np.log10(365 * day_to_seconds), color='grey', linestyle='--', lw=2., zorder=-2)
    axis.annotate('1 day', xy=(x_label_pos, np.log10(1 * day_to_seconds) + y_label_offset), ha='left', va='center',
                  color='grey')
    axis.annotate('1 month', xy=(x_label_pos, np.log10(30 * day_to_seconds) + y_label_offset), ha='left', va='center',
                  color='grey')
    axis.annotate('1 year', xy=(x_label_pos, np.log10(365 * day_to_seconds) + y_label_offset), ha='left', va='center',
                  color='grey')
    '''plt.annotate('4 months', xy=(1 / 120 - 0.0015, y_coord_pos + 0.0005), ha='center', va='bottom', rotation=90,
                  color='grey')'''

    axis.indicate_inset_zoom(ax_inset, edgecolor="black")
    axis.set_xlim([11, 24])  # plt.xlim([11, 22])
    axis.set_ylim([-2, 8])
    axis.set_xlabel('log$_{10}$($M_0$)')
    axis.set_ylabel('log$_{10}$(duration [s])')
    mwax.set_xlabel('$M_w$')
    if legend:
        axis.legend()


def _gutenberg_richter(axis: Axes, se: SlowSlipEventExtractor, refine_durations: bool, legend: bool, N_BINS: int = 30,
                       compute_b_value=False):
    mw_bin_list, num_event_list = [], []
    time_array = se.ddh.corrected_time
    b_values = []
    x_gr, y_gr = [], []
    # gr_bounds = {0.07: (6., 6.5), 0.3: (6., 6.5), 0.5: (5.8, 6.2), 0.7: (5.7, 6.2)}
    gr_bounds = {0.07: (5.8, 6.3), 0.3: (5.8, 6.3), 0.5: (5.8, 6.3), 0.7: (5.8, 6.3)}

    for slip_thresh in se.slip_thresholds:
        event_moment_list = np.array([mw_to_mo(mw) for mw in se.get_magnitude_events(slip_thresh, refine_durations)])

        valid_mo_idx = np.where(event_moment_list > 0.)[0]
        mw_array = np.array([mo_to_mw(mo) for mo in event_moment_list[valid_mo_idx]])
        abs_dates_events = np.array(se.get_event_date_idx(slip_thresh, refine_durations))[valid_mo_idx]

        mw_bins = np.linspace(min(mw_array), max(mw_array), num=N_BINS)

        num_events = [len([mag for mag in mw_array if mag >= bin]) for bin in mw_bins]

        time_span_years = time_array[-1] - time_array[0]
        num_events = np.array(num_events) / time_span_years  # obtain the yearly rate

        axis.scatter(mw_bins, num_events, edgecolors=matplotlib.colors.colorConverter.to_rgba('black', alpha=.5),
                     label=f'{slip_thresh}')  # label=f'thresh: {slip_thresh} mm/day'
        mw_bin_list.append(mw_bins)
        num_event_list.append(num_events)
        if compute_b_value:
            log_counts = np.log10(num_events)
            valid_mw_filter = np.where((mw_bins >= gr_bounds[slip_thresh][0]) & (mw_bins <= gr_bounds[slip_thresh][1]))[
                0]
            slope, intercept = np.polyfit(mw_bins[valid_mw_filter], log_counts[valid_mw_filter], 1)
            b_value = -slope
            b_values.append(b_value)
            x_gr.append(mw_bins[valid_mw_filter])
            y_gr.append(mw_bins[valid_mw_filter] * slope + intercept)

    axis.set_yscale('log')
    axis.set_xlabel('$M_w$')
    axis.set_ylabel('Number of events per year\nwith magnitude $> M_w$')

    if compute_b_value:
        y_offset_gr = [1.4, 1.25, 0.7, 0.5]
        x_pos = [6.5, 6.5, 6., 5.9]
        y_pos = [4, 2, 2, 0.5]
        x_offset = [-0.1, 0.3, -0.5, -0.5]
        y_offset = [-1, -1.6, 0.1, 0.35]
        b_val_rotation = [-b_values[0], 0, 0, 0]
        colors = ['C0', 'C1', 'C2', 'C3']
        lines = []
        for i, slip_thresh in enumerate(se.slip_thresholds):
            l,=axis.plot(x_gr[i], y_offset_gr[i] * 10 ** (y_gr[i]), linestyle='--', linewidth=2.5)
            '''axis.annotate(f'b={round(b_values[i], 2)}', xy=(x_pos[i] + x_offset[i], y_offset[i] + (y_pos[i])),
                          ha='center', va='center', rotation=np.degrees(np.arctan(b_val_rotation[i])), color=colors[i])'''
            lines.append(l)

    if legend:
        def legend_title_left(leg):
            c = leg.get_children()[0]
            title = c.get_children()[0]
            hpack = c.get_children()[1]
            c._children = [hpack]
            hpack._children = [title] + hpack.get_children()
        # legend = axis.legend(title='Threshold [mm/day]', ncol=4, bbox_to_anchor=(0.3, 1.01), title_fontproperties={'weight':'bold'})
        '''legend._legend_box.align = 'left'
        legend._legend_box.set_width(430)
        legend._legend_box.set_height(50)
        legend.get_title().set_position((-220, -25))'''
        # legend_title_left(legend)
        if compute_b_value:
            b_val_leg = [f'b={round(bval, 2)}' for bval in b_values]
            legend_b_val = axis.legend(lines, b_val_leg)

            #axis.add_artist(legend)
            #axis.add_artist(legend_b_val)


def _mo_area_scaling(axis: Axes, se: SlowSlipEventExtractor, refine_durations: bool, legend: bool):
    # accounting for rectangular dislocations
    C = 1  # shape factor, from Kanamori and Anderson, 1975
    aspect_ratio = 1  # L/W
    C_tilde = 1 / (C * np.sqrt(aspect_ratio))  # all shape factors
    size_factor = 1e6  # conversion from km^2 to mm^2
    mwax = axis.secondary_xaxis('top', functions=(logmo_to_mw, mw_to_logmo))
    mo_dummy_array = np.array([14, 21])

    min_sd_eq, max_sd_eq = 1., 10.  # 0.01, 0.1  # MPa
    min_sd_gao, max_sd_gao = 0.01, 0.1  # MPa
    min_sd_intercept_eq = -2 / 3 * np.log10(min_sd_eq * 1e6) - 2 / 3 * np.log10(C_tilde)
    max_sd_intercept_eq = -2 / 3 * np.log10(max_sd_eq * 1e6) - 2 / 3 * np.log10(C_tilde)
    min_sd_intercept_gao = -2 / 3 * np.log10(min_sd_gao * 1e6) - 2 / 3 * np.log10(C_tilde)
    max_sd_intercept_gao = -2 / 3 * np.log10(max_sd_gao * 1e6) - 2 / 3 * np.log10(C_tilde)
    '''axis.fill_between(mo_dummy_array,
                      straight_line(mo_dummy_array, 2 / 3, min_sd_intercept_eq) - np.log10(size_factor),
                      straight_line(mo_dummy_array, 2 / 3, max_sd_intercept_eq) - np.log10(size_factor), color='C3',
                      alpha=0.5)
    axis.fill_between(mo_dummy_array,
                      straight_line(mo_dummy_array, 2 / 3, min_sd_intercept_gao) - np.log10(size_factor),
                      straight_line(mo_dummy_array, 2 / 3, max_sd_intercept_gao) - np.log10(size_factor), color='C0',
                      alpha=0.5)'''

    for slip_thresh in se.slip_thresholds:
        event_moment_list = np.array([mw_to_mo(mw) for mw in se.get_magnitude_events(slip_thresh, refine_durations)])
        event_areas = se.get_area_events(slip_thresh, refine_durations)

        valid_mo_idx = np.where(event_moment_list > 0.)[0]
        event_moment_list = event_moment_list[valid_mo_idx]
        event_areas = event_areas[valid_mo_idx]
        # for axx in (axis, ax_inset):
        axis.scatter(np.log10(event_moment_list), np.log10(event_areas),
                     edgecolors=matplotlib.colors.colorConverter.to_rgba('black', alpha=.5),
                     label=f'thresh: {slip_thresh} mm/day')

    # x_label_position = 20.5
    x_label_position = 15.5
    '''_annotate_line(axis, x_label_position, 2 / 3, min_sd_intercept_eq - np.log10(size_factor),
                   f'${int(min_sd_eq)} MPa$', color='k', y_label_shift=.2, positive_y_shift=True, x_axis_log=False,
                   y_axis_log=False, x_label_shift=0, positive_x_shift=False, screen_space=False, forced_angle=2 / 3)
    _annotate_line(axis, x_label_position, 2 / 3, max_sd_intercept_eq - np.log10(size_factor),
                   f'${int(max_sd_eq)} MPa$', color='k', y_label_shift=.2, positive_y_shift=True, x_axis_log=False,
                   y_axis_log=False, x_label_shift=0, positive_x_shift=False, screen_space=False, forced_angle=2 / 3)'''
    # draw some iso-stress-drop lines
    # sd_values_isolines = np.array([1, 10, 100]) * 1e3  # 1 KPa -> (cf. Michel et al., 2019), 10 KPa, 100 KPa -> (cf. Gao et al., 2012)
    sd_values_isolines = np.array(
        [1]) * 1e3  # 1 KPa -> (cf. Michel et al., 2019), 10 KPa, 100 KPa -> (cf. Gao et al., 2012)
    for sd_value in sd_values_isolines:
        sd_intercept = -2 / 3 * np.log10(sd_value) - 2 / 3 * np.log10(C_tilde)
        # for axx in (axis, ax_inset):
        axis.plot(mo_dummy_array, straight_line(mo_dummy_array, 2 / 3, sd_intercept - np.log10(size_factor)),
                  linestyle='--', color='k', alpha=.5, zorder=-1)
        _annotate_line(axis, x_label_position, 2 / 3, sd_intercept - np.log10(size_factor),
                       f'${int(sd_value * 1e-3)} KPa$', color='k', y_label_shift=.3, positive_y_shift=False,
                       x_axis_log=False, y_axis_log=False, x_label_shift=0, positive_x_shift=False, screen_space=False,
                       forced_angle=2 / 3)

    # axis.indicate_inset_zoom(ax_inset, edgecolor="black")
    axis.set_xlabel('log$_{10}$($M_0$)')
    axis.set_ylabel('log$_{10}$(area [km$^2$])')
    mwax.set_xlabel('$M_w$')
    ylim = axis.get_ylim()
    axis.set_ylim([1.5, ylim[1]])
    axis.set_xlim([14.5, 20])
    if legend:
        axis.legend()
    plt.margins(x=0)
    plt.margins(y=0)


def scaling_laws(se: SlowSlipEventExtractor, refine_durations: bool, dpi: int = 100,
                 compute_b_value=False):
    # fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=dpi)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=dpi)  # 7:2 ratio

    _mo_duration_scaling(axes[0], se, refine_durations, legend=False)
    _gutenberg_richter(axes[1], se, refine_durations, legend=True, compute_b_value=compute_b_value)
    _mo_area_scaling(axes[2], se, refine_durations, legend=False)

    '''for ax, label in zip(axes, ['A', 'B', 'C']):
        ax.text(-0.1, 1.15, label, transform=ax.transAxes,
                fontsize=20, fontweight=1000, va='top', ha='right')'''
    plt.tight_layout()
    plt.savefig('figures/scaling_laws.pdf', bbox_inches='tight')
    plt.close(fig)
