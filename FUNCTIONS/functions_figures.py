import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d, interp2d
from scipy.signal import detrend
from scipy.stats import linregress
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

from FUNCTIONS.functions_slab import UTM_GEO
from sse_extraction.SlowSlipEventExtractor import SlowSlipEventExtractor
from utils.colormaps import custom_blues_colormap
from utils.fourier_transform import custom_fft
from utils.geo_functions import mo_to_mw, logmo_to_mw, mw_to_logmo, mw_to_mo
from utils.math_utils import straight_line
from utils.slab_plot_functions import init_basemap_cascadia, isodepth_label_fmt
from utils.slab_utils import read_depth_from_slab2
from utils.slip_evolution_functions import fit_slip_evolution, triangular_slip_evolution


def _annotate_line(axis, label_x_pos, slope, intercept, text, positive_y_shift=True, positive_x_shift=True, color='k',
                   x_label_shift=1.5, y_label_shift=1.1, x_axis_log=True, y_axis_log=True, eps=.01, screen_space=True,
                   transform_rotates_text=False, forced_angle=None):
    def ylabel_pos(x, slope, intercept, x_axis_log=x_axis_log):
        x_val = np.log10(x) if x_axis_log else x
        pos = x_val * slope + intercept
        return pos

    label_pos = ylabel_pos(label_x_pos, slope, intercept)
    # np.log10(label_freq_pos) * global_avg_slope + global_avg_intercept + global_std_intercept

    # label xy and rotation angle need to be recomputed because of the log-log transformation
    if y_axis_log:
        xy = np.array((label_x_pos, 10 ** label_pos))
        p = (label_x_pos + eps, 10 ** ylabel_pos(label_x_pos + eps, slope, intercept))  # gradient
    else:
        xy = np.array((label_x_pos, label_pos))
        p = (label_x_pos + eps, ylabel_pos(label_x_pos + eps, slope, intercept))

    p = axis.transData.transform_point(p)
    pa = axis.transData.transform_point(xy)

    screen_angle = np.arctan2(p[1] - pa[1], p[0] - pa[0])

    sign_y_shift = 1 if positive_y_shift else -1
    sign_x_shift = 1 if positive_x_shift else -1

    if x_axis_log:
        # Shift x in logarithmic coordinates
        shifted_x = xy[0] * (x_label_shift ** sign_x_shift)
    else:
        # Shift x in linear coordinates
        shifted_x = xy[0] + (x_label_shift * sign_x_shift)

    if y_axis_log:
        # Shift y in logarithmic coordinates
        shifted_y = xy[1] * (y_label_shift ** sign_y_shift)
    else:
        # Shift y in linear coordinates
        shifted_y = xy[1] + (y_label_shift * sign_y_shift)

    # Combine the shifted coordinates
    shifted_xy = np.array([shifted_x, shifted_y])

    rotation_angle = np.degrees(screen_angle) if screen_space else np.degrees(np.arctan(slope))
    if forced_angle is not None:
        rotation_angle = np.degrees(np.arctan(forced_angle))

    axis.annotate(text, xy=shifted_xy, ha='center', va='center', rotation=rotation_angle, color=color,
                  transform_rotates_text=transform_rotates_text)


def spectra_compilation_mo_rate(sse_info_thresh, mo_rates, time, slip_thresholds, slip, area, shear_modulus, sf=1,
                                num_freq_bins=10, plot_corner_freq=False):
    for thresh in slip_thresholds:
        figure, axis = plt.subplots(1, 1, figsize=(8, 7), dpi=300, sharex='all', sharey='all')

        mo_rate = np.sum(mo_rates, axis=1)  # total moment release

        freq, fft = custom_fft(mo_rate, sf, len(time), axis=0, pos_freq_only=True)

        tot_mo_rate = np.sum(mo_rate)
        fft = fft ** 2 / tot_mo_rate ** 2  # we look at the power, as Hawthorne and Bartlow (2018)

        plt.plot(freq, fft, color='k', lw=2.)

        min_global_freq = 1 / 100  # day^-1, minimum frequency for global slope calculation
        freq_mask = freq > min_global_freq

        result = linregress(np.log10(freq[freq_mask]), np.log10(fft[freq_mask]))
        global_slope, global_intercept = result.slope, result.intercept

        event_moment_list, event_duration_list, event_area_list, slip_event_list, patch_idx_list, date_list, mo_rate_list, slip_rate_list = \
            sse_info_thresh[thresh]
        event_moment_list, event_duration_list = np.array(event_moment_list), np.array(event_duration_list)
        mw_list = [mo_to_mw(mo) if mo != 0 else np.nan for mo in event_moment_list]
        min_mw, max_mw = np.nanmin(mw_list), np.nanmax(mw_list)
        cmap = matplotlib.cm.get_cmap('turbo')
        norm = Normalize(vmin=min_mw, vmax=max_mw)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy array for the ScalarMappable
        fft_event_list = []

        duration_list = []
        freq_list = []
        valid_mw_list = []
        slip[slip < thresh] = 0.
        slip_potency = area * slip * 1e-03  # slip converted to meters
        for i, mo_rate_event in enumerate(mo_rate_list):
            start_idx, end_idx = date_list[i]
            ev_dur = end_idx - start_idx + 1
            if start_idx - ev_dur < 0 or end_idx + ev_dur > len(mo_rates):
                continue
            ext_slip_potency = slip_potency[start_idx - 0:end_idx + 1 + 0, patch_idx_list[i]].copy()
            mo_rate_event = ext_slip_potency * shear_modulus

            threshold = 0.95 * mo_rate_event.shape[0]
            zero_count = np.sum(mo_rate_event == 0., axis=0)
            filtered_mo_rate_event = mo_rate_event[:, zero_count <= threshold]  # filter out zero-valued patches

            filtered_mo_rate_event = np.sum(filtered_mo_rate_event, axis=1)  # total mo rate (for all patches)

            # zero padding only for visualization
            n_padding = 3 * len(filtered_mo_rate_event)
            padded_mo_rate_event = np.pad(filtered_mo_rate_event, (0, n_padding), 'constant', constant_values=0.)
            padded_freq_event, padded_fft_event = custom_fft(padded_mo_rate_event, sf, len(padded_mo_rate_event),
                                                             axis=0,
                                                             pos_freq_only=True)
            padded_fft_event = padded_fft_event ** 2 / tot_mo_rate ** 2  # we look at the power, as Hawthorne and Bartlow (2018)

            freq_event, fft_event = custom_fft(filtered_mo_rate_event, sf, len(filtered_mo_rate_event), axis=0,
                                               pos_freq_only=True)

            fft_event = fft_event ** 2 / tot_mo_rate ** 2  # we look at the power, as Hawthorne and Bartlow (2018)

            if fft_event.shape[0] == 0:
                continue

            approx_corner_freq = 1 / len(filtered_mo_rate_event)
            valid_event_freq_mask = freq_event > approx_corner_freq
            if freq_event[valid_event_freq_mask].shape[0] == 0:
                continue
            result = linregress(np.log10(freq_event[valid_event_freq_mask]), np.log10(fft_event[valid_event_freq_mask]))
            slope, intercept = result.slope, result.intercept
            if np.isnan(slope):
                continue

            plt.plot(freq_event, fft_event, color=cmap(norm(mw_list[i])), lw=2., alpha=.75, zorder=10 + 1 / mw_list[i])
            # plot dashed zero-padded spectrum until the approximate corner frequency
            plt.plot(padded_freq_event[padded_freq_event <= approx_corner_freq],
                     padded_fft_event[padded_freq_event <= approx_corner_freq], color=cmap(norm(mw_list[i])),
                     lw=2., alpha=.3, zorder=10 + 1 / mw_list[i], linestyle='--')

            '''if mw_list[i] > 6.2:
                plt.plot(freq_event[0], fft_event[0], 'x', color='black', zorder=1000)
            else:
                plt.plot(freq_event[0], fft_event[0], 'o', color='black', zorder=1000)'''
            fft_event_list.append(fft_event)
            duration_list.append(len(filtered_mo_rate_event))
            freq_list.append(freq_event)
            valid_mw_list.append(mw_list[i])

        # evaluate the sum of the events' spectra (as Hawthorne and Bartlow, 2018)
        longest_freq_array = freq_list[np.argmax(duration_list)]
        bin_edges = np.histogram_bin_edges(longest_freq_array, bins=num_freq_bins)
        binned_amp_sum = np.zeros(len(bin_edges) - 1)

        for n_ev, fft_event in enumerate(fft_event_list):
            freq_event = freq_list[n_ev]
            for i in range(len(bin_edges) - 1):
                bin_mask = (freq_event >= bin_edges[i]) & (freq_event < bin_edges[i + 1])
                binned_amp_sum[i] += np.sum(fft_event[bin_mask])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # plt.plot(bin_edges[:-1], binned_amp_sum, color='g', lw=2.)
        plt.plot(bin_centers, binned_amp_sum, color='g', lw=2.)

        binned_amp_sum = np.zeros(len(bin_edges) - 1)
        for n_ev, fft_event in enumerate(fft_event_list):
            freq_event = freq_list[n_ev]
            hist, _ = np.histogram(freq_event, bins=bin_edges, weights=fft_event)
            binned_amp_sum += hist

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, binned_amp_sum, color='grey', lw=2.)

        plt.plot([], [], color='k', lw=2., label='15-year-long sequence')
        plt.plot([], [], color='r', lw=2., label='Single events')

        '''global_spectrum_y_offset = 10 ** 0.1
        plt.plot(freq[freq_mask],
                 10 ** (global_slope * np.log10(freq[freq_mask]) + global_intercept + global_spectrum_y_offset), '--',
                 lw=2., color='k')'''

        plt.xscale('log')
        plt.yscale('log')

        cbar = plt.colorbar(sm, ax=axis)
        cbar.ax.set_ylabel('$M_w$', rotation=270, labelpad=20)

        y_coord_pos = axis.get_ylim()[0]
        plt.axvline(x=1 / 365, color='grey', linestyle='--', lw=2., zorder=-1)
        plt.axvline(x=1 / 120, color='grey', linestyle='--', lw=2., zorder=-1)
        axis.annotate('1 year', xy=(1 / 365 - 0.0005, y_coord_pos * 1.2), ha='center', va='bottom', rotation=90,
                      color='grey')
        axis.annotate('4 months', xy=(1 / 120 - 0.0015, y_coord_pos * 1.2), ha='center', va='bottom', rotation=90,
                      color='grey')

        plt.xlabel('Frequency [day$^{-1}$]')
        plt.ylabel('Normalized $\dot M_0$ power spectrum $|\widehat \dot M_0(f)|^2$ / $(\widebar \dot M_0)^2$')

        '''label_freq_pos = freq[np.abs(freq - 1 / 20) < 0.001][0]  # center the text at about 1/20 days^-1
        _annotate_line(axis, label_freq_pos, global_slope, global_intercept + global_spectrum_y_offset,
                       f'$\propto f^{{-n}}, n={round(abs(global_slope), 2)}$')'''

        # perform linear fitting of corner frequencies, suppose a knee at about Mw 6.2 to have two separate regressions
        if plot_corner_freq:
            mw_thresh_bounded = 6.2
            # we take the first value of the fft, assuming that f_c ~ 1/T
            corner_frequencies = np.array([frq[0] for frq in freq_list])
            if thresh == 0.07:
                y_fc = np.array([ft[0] for ft in fft_event_list])  # fft values evaluated at corner freq
                mw_filter_bounded = np.array(valid_mw_list) > mw_thresh_bounded

                mw_filter_unbounded = np.logical_and(np.array(valid_mw_list) > 5.2,
                                                     np.array(valid_mw_list) < mw_thresh_bounded)
                result = linregress(np.log10(corner_frequencies[mw_filter_unbounded]),
                                    np.log10(y_fc[mw_filter_unbounded]))
                unbounded_slope, unbounded_intercept = result.slope, result.intercept
                result = linregress(np.log10(corner_frequencies[mw_filter_bounded]), np.log10(y_fc[mw_filter_bounded]))
                bounded_slope, bounded_intercept = result.slope, result.intercept
                print('bounded slope', bounded_slope)
                print('unbounded slope', unbounded_slope)
                plt.plot(corner_frequencies[mw_filter_unbounded],
                         10 ** (unbounded_slope * np.log10(
                             corner_frequencies[mw_filter_unbounded]) + unbounded_intercept - .6),
                         '--',
                         color='black', zorder=1000)
                plt.plot(corner_frequencies[mw_filter_bounded],
                         10 ** (bounded_slope * np.log10(corner_frequencies[mw_filter_bounded]) + bounded_intercept),
                         '--',
                         color='black', zorder=1000)
            # print('unbounded_slope', unbounded_slope)
            # print('bounded_slope', bounded_slope)

        # plt.savefig(f'figures/mo_rate_spectra/mo_rate_spectrum_{thresh}.pdf', bbox_inches='tight')
        plt.savefig(f'figures/mo_rate_spectra/mo_rate_spectrum_{thresh}.pdf', bbox_inches='tight')
        # plt.show()
        plt.close(figure)


def mo_rate_stack_asymmetry_patchwise(sse_info_thresh, slip_thresholds, n_dur_bins=5, show_fit=False,
                                      show_individual_mo=True,
                                      align_start=False, rescale_zero_y=False, base_dir='figures'):
    local_dir = os.path.join(base_dir, 'asymmetry')
    os.makedirs(local_dir, exist_ok=True)
    bins_folder = os.path.join(local_dir, f'{n_dur_bins}_bins')
    os.makedirs(bins_folder, exist_ok=True)

    for thresh in slip_thresholds:
        thresh_folder = f'thresh_{thresh}'
        local_subfolder = os.path.join(bins_folder, thresh_folder)
        os.makedirs(local_subfolder, exist_ok=True)
        event_moment_list, event_duration_list, event_area_list, slip_event_list, patch_idx_list, date_list, mo_rate_list, slip_rate_list = \
            sse_info_thresh[thresh]

        mo_rate_list_all_patches = [mo_rate_list[i][:, j] for i in range(len(mo_rate_list)) for j in
                                    range(len(mo_rate_list[i]))]

        trimmed_mo_rates = []  # remove heading/trailing zeros
        for i in range(len(mo_rate_list_all_patches)):
            trimmed_mo = np.trim_zeros(mo_rate_list_all_patches[i])
            valid_mo = np.concatenate(([0.], trimmed_mo, [0.]))  # add zeros to avoid edge effects
            if len(valid_mo) > 2:  # remove events with less than 2 points
                trimmed_mo_rates.append(valid_mo)
        mo_rate_list_all_patches = trimmed_mo_rates.copy()

        mw_all_patches = [mo_to_mw(np.sum(mo_rate)) if np.sum(mo_rate) != 0 else np.nan for mo_rate in
                          mo_rate_list_all_patches]
        mw_all_patches = np.array(mw_all_patches)
        mw_all_patches = mw_all_patches[~np.isnan(mw_all_patches)]

        dur_all_patches = [len(mo_rate) for mo_rate in mo_rate_list_all_patches]

        # create bins with approximately the same number of samples
        dur_bins = np.percentile(dur_all_patches, np.linspace(0, 100, n_dur_bins + 1))
        # mw_bins = np.percentile(mw_all_patches, np.linspace(0, 100, n_mw_bins + 1))
        # bin_edges = np.histogram_bin_edges(mw_bins, bins=n_mw_bins)

        # mw-related colorbar is computed based on mean Mo per bin
        mean_mo_per_bin = []
        for i in range(len(dur_bins) - 1):
            dur_bin = (dur_bins[i], dur_bins[i + 1])
            bin_idx = np.where((dur_all_patches >= dur_bin[0]) & (dur_all_patches < dur_bin[1]))[0]
            mean_mo_per_bin.append(np.mean([np.sum(mo_rate_list_all_patches[j]) for j in bin_idx]))
        mean_mw_per_bin = [mo_to_mw(mo) for mo in mean_mo_per_bin]

        figure, axis = plt.subplots(1, 1, figsize=(10, 7), dpi=300, sharex='all', sharey='all')

        base_cmap = matplotlib.cm.Blues
        cmap = LinearSegmentedColormap.from_list(
            'Blues_darker', base_cmap(np.linspace(0.2, 1, 256)))
        colors = cmap(np.linspace(0, 1, len(mean_mw_per_bin)))
        norm = Normalize(vmin=np.min(mean_mw_per_bin), vmax=np.max(mean_mw_per_bin))
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        acc_times, dec_times, mo_amplitudes, avg_moments = [], [], [], []
        for i in range(len(dur_bins) - 1):
            dur_bin = (dur_bins[i], dur_bins[i + 1])
            print('bin:', dur_bin)
            bin_idx = np.where((dur_all_patches >= dur_bin[0]) & (dur_all_patches < dur_bin[1]))[0]
            print(len(bin_idx), 'events with duration in', dur_bin)

            if len(bin_idx) == 0 or dur_bins[i] < 4.:  # maybe it could be done in a better way...
                acc_times.append(np.nan)  # add nan to skip a point in the plot (dur_bins is left untouched)
                dec_times.append(np.nan)
                mo_amplitudes.append(np.nan)
                avg_moments.append(np.nan)
                continue

            max_duration_bin = int(dur_bins[i + 1])
            max_time_array_in_bin = np.linspace(0, 1, max_duration_bin)  # in [0,1] for correct resampling
            max_time_array_in_bin_actual = np.linspace(0, max_duration_bin,
                                                       max_duration_bin)  # to plot the right duration
            binned_stack, mo_list_stack = [], []
            for idx in bin_idx:
                mo_len = len(mo_rate_list_all_patches[idx])
                f_int = interp1d(np.linspace(0, 1, mo_len), mo_rate_list_all_patches[idx], kind='linear')
                upsampled_mo_rate = f_int(max_time_array_in_bin)
                n_smooth_kernel = 7
                if max_duration_bin > 10:  # perform smoothing on samples that are long enough
                    avg_smoothing_kernel = np.ones(n_smooth_kernel) / n_smooth_kernel
                    upsampled_mo_rate = np.convolve(upsampled_mo_rate, avg_smoothing_kernel, mode='same')

                # amax = np.argmax(upsampled_mo_rate) / max_duration_full
                time_shift = 0 if align_start else np.argmax(upsampled_mo_rate)
                # upsampled_mo_rate = upsampled_mo_rate / np.max(upsampled_mo_rate)
                if show_individual_mo:
                    plt.plot(max_time_array_in_bin_actual - time_shift, upsampled_mo_rate, alpha=.05,
                             color=cmap(norm(mw_all_patches[idx])))
                binned_stack.append(upsampled_mo_rate)
                mo_list_stack.append(np.sum(mo_rate_list_all_patches[idx]))

            # binned_stack = np.array(binned_stack)
            avg_moment_stack = np.mean(mo_list_stack)
            binned_stack = np.nansum(binned_stack, axis=0) / len(binned_stack)
            # binned_stack = binned_stack / np.max(binned_stack)
            '''if rescale_stack_mo:
                binned_stack = binned_stack * avg_moment_stack'''

            print('mo recomputed', mo_to_mw(np.sum(binned_stack)), mean_mw_per_bin)

            time_shift_stack = 0 if align_start else np.argmax(binned_stack)
            y_shift_stack = binned_stack[0] if rescale_zero_y else 0
            print('shift', y_shift_stack)
            # plt.plot(max_time_array - amax, binned_stack, color=colors[i], lw=2., label='weighted_stack')
            plt.plot(max_time_array_in_bin_actual - time_shift_stack, binned_stack - y_shift_stack,
                     color=colors[i], lw=2.)
            # Fit slip evolution function to stacked Mo rates
            opt_params = fit_slip_evolution(max_time_array_in_bin_actual, binned_stack, var_on_peak=1,
                                            slip_ev_fcn_signature=triangular_slip_evolution)
            A_fit, T0_fit, T1_fit, T2_fit = opt_params

            acc_times.append(T1_fit - T0_fit)
            mo_amplitudes.append(A_fit)
            avg_moments.append(avg_moment_stack)
            dec_times.append(T2_fit - T1_fit)

            slip_ev_fit = triangular_slip_evolution(max_time_array_in_bin_actual, *opt_params)
            if show_fit:
                plt.plot(max_time_array_in_bin_actual - time_shift_stack, slip_ev_fit - y_shift_stack, '--',
                         color=colors[i], lw=2.)
            # plt.show()
        fit_str = '_fit' if show_fit else ''
        n_bin_str = f'_{n_dur_bins}_bins'
        thresh_str = f'_thresh_{thresh}'

        cbar = plt.colorbar(sm, ax=axis)
        cbar.ax.set_ylabel('Moment magnitude (Mw)', rotation=270, labelpad=25)

        plt.ylabel('Moment rate function [$N\cdot m \cdot d^{-1}$]')

        if align_start:
            plt.xlabel('Duration [days]')
        else:
            plt.xlabel('Duration (relative to peak) [days]')
        plt.savefig(f'{local_subfolder}/patchwise_moment_rate_evolution{fit_str}{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        acc_times, mo_amplitudes, dec_times = np.array(acc_times), np.array(mo_amplitudes), np.array(dec_times)
        avg_moments = np.array(avg_moments)
        acc_times[acc_times > 100] = np.nan  # remove extreme values
        acc_times[acc_times < 0] = np.nan  # remove extreme values
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Acceleration time [days]')
        plt.savefig(f'{local_subfolder}/fitted_acc_time_vs_duration{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.plot(avg_moments, acc_times)

        '''def exponential_function(x, a, b, c):
            return a * np.exp(x * b) + c
        popt, pcov = curve_fit(exponential_function, avg_moments, acc_times, p0=(1, 1e-16, 1), nan_policy='omit')
        x_fit = np.linspace(np.nanmin(avg_moments), np.nanmax(avg_moments), 100)
        plt.plot(x_fit, exponential_function(x_fit, *popt), '--')'''

        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Acceleration time [days]')
        plt.savefig(f'{local_subfolder}/fitted_acc_time_vs_mo{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], mo_amplitudes / acc_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Moment acceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/fitted_mo_acc_vs_duration{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.plot(avg_moments, mo_amplitudes / acc_times)
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Moment acceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/fitted_mo_acc_vs_mo{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        # deceleration
        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], dec_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Deceleration time [days]')
        plt.savefig(f'{local_subfolder}/fitted_dec_time_vs_duration{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.plot(avg_moments, dec_times)
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Deceleration time [days]')
        plt.savefig(f'{local_subfolder}/fitted_dec_time_vs_mo{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.plot(avg_moments, (acc_times) / (acc_times + dec_times))
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Normalized acceleration time [%]')
        plt.ylim([0, 1])
        plt.savefig(f'{local_subfolder}/fitted_accnorm_time_vs_mo{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], - mo_amplitudes / dec_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Moment deceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/fitted_mo_dec_vs_duration{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.plot(avg_moments, - mo_amplitudes / dec_times)
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Moment deceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/fitted_mo_dec_vs_mo{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()


def mo_rate_stack_asymmetry_eventwise(sse_info_thresh, slip_thresholds, new_duration_dict, n_dur_bins=5, show_fit=False,
                                      show_individual_mo=True, align_start=False, rescale_zero_y=False,
                                      base_dir='figures', refine_durations=True):
    local_dir = os.path.join(base_dir, 'asymmetry')
    os.makedirs(local_dir, exist_ok=True)
    bins_folder = os.path.join(local_dir, f'{n_dur_bins}_bins')
    os.makedirs(bins_folder, exist_ok=True)

    for thresh in slip_thresholds:
        thresh_folder = f'thresh_{thresh}'
        local_subfolder = os.path.join(bins_folder, thresh_folder)
        os.makedirs(local_subfolder, exist_ok=True)
        event_moment_list, event_duration_list, event_area_list, slip_event_list, patch_idx_list, date_list, mo_rate_list, slip_rate_list = \
            sse_info_thresh[thresh]

        if refine_durations:
            mo_rate_list_all_events = []
            for i, mr in enumerate(mo_rate_list):
                idx = new_duration_dict[thresh][i]
                # new_start = date_list[i][0] + idx[0]
                # new_end = date_list[i][0] + idx[1]
                new_start = idx[0]
                new_end = idx[1]
                mo_rate_list_all_events.append(np.sum(mr[new_start:new_end], axis=1))
        else:
            mo_rate_list_all_events = [np.sum(mo_rate_list[i], axis=1) for i in range(len(mo_rate_list))]

        '''for i in range(len(mo_rate_list_all_events)):
            plt.plot(mo_rate_list_all_events[i])
            plt.title(len(mo_rate_list_all_events[i]))
            plt.show()'''

        '''trimmed_mo_rates = []  # remove heading/trailing zeros
        for i in range(len(mo_rate_list_all_patches)):
            trimmed_mo = np.trim_zeros(mo_rate_list_all_patches[i])
            valid_mo = np.concatenate(([0.], trimmed_mo, [0.]))  # add zeros to avoid edge effects
            if len(valid_mo) > 2:  # remove events with less than 2 points
                trimmed_mo_rates.append(valid_mo)
        mo_rate_list_all_patches = trimmed_mo_rates.copy()'''

        '''mw_all_events = [mo_to_mw(np.sum(mo_rate)) if np.sum(mo_rate) != 0 else np.nan for mo_rate in
                          mo_rate_list_all_events]
        mw_all_events = np.array(mw_all_events)
        mw_all_events = mw_all_events[~np.isnan(mw_all_events)]'''

        dur_all_events = [len(mo_rate) for mo_rate in mo_rate_list_all_events]

        # create bins with approximately the same number of samples
        dur_bins = np.percentile(dur_all_events, np.linspace(0, 100, n_dur_bins + 1))
        # mw_bins = np.percentile(mw_all_events, np.linspace(0, 100, n_dur_bins + 1))
        # bin_edges = np.histogram_bin_edges(mw_bins, bins=n_mw_bins)

        # mw-related colorbar is computed based on mean Mo per bin
        mean_mo_per_bin = []
        for i in range(len(dur_bins) - 1):
            dur_bin = (dur_bins[i], dur_bins[i + 1])
            bin_idx = np.where((dur_all_events >= dur_bin[0]) & (dur_all_events < dur_bin[1]))[0]
            mean_mo_per_bin.append(np.mean([np.sum(mo_rate_list_all_events[j]) for j in bin_idx]))
        mean_mw_per_bin = [mo_to_mw(mo) for mo in mean_mo_per_bin]

        figure, axis = plt.subplots(1, 1, figsize=(10, 7), dpi=300, sharex='all', sharey='all')

        base_cmap = matplotlib.cm.Blues
        cmap = LinearSegmentedColormap.from_list(
            'Blues_darker', base_cmap(np.linspace(0.2, 1, 256)))
        colors = cmap(np.linspace(0, 1, len(mean_mw_per_bin)))
        norm = Normalize(vmin=np.nanmin(mean_mw_per_bin), vmax=np.nanmax(mean_mw_per_bin))
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        acc_times, dec_times, mo_amplitudes, avg_moments = [], [], [], []
        for i in range(len(dur_bins) - 1):
            dur_bin = (dur_bins[i], dur_bins[i + 1])
            print('bin:', dur_bin)
            bin_idx = np.where((dur_all_events >= dur_bin[0]) & (dur_all_events < dur_bin[1]))[0]
            print(len(bin_idx), 'events with duration in', dur_bin)

            if len(bin_idx) == 0 or dur_bins[i] < 4.:  # maybe it could be done in a better way...
                acc_times.append(np.nan)  # add nan to skip a point in the plot (dur_bins is left untouched)
                dec_times.append(np.nan)
                mo_amplitudes.append(np.nan)
                avg_moments.append(np.nan)
                continue

            max_duration_bin = int(dur_bins[i + 1])
            max_time_array_in_bin = np.linspace(0, 1, max_duration_bin)  # in [0,1] for correct resampling
            max_time_array_in_bin_actual = np.linspace(0, max_duration_bin,
                                                       max_duration_bin)  # to plot the right duration
            binned_stack, mo_list_stack = [], []
            for idx in bin_idx:
                mo_len = len(mo_rate_list_all_events[idx])
                f_int = interp1d(np.linspace(0, 1, mo_len), mo_rate_list_all_events[idx], kind='linear')
                upsampled_mo_rate = f_int(max_time_array_in_bin)
                n_smooth_kernel = 7
                if max_duration_bin > 10:  # perform smoothing on samples that are long enough
                    avg_smoothing_kernel = np.ones(n_smooth_kernel) / n_smooth_kernel
                    upsampled_mo_rate = np.convolve(upsampled_mo_rate, avg_smoothing_kernel, mode='same')

                # amax = np.argmax(upsampled_mo_rate) / max_duration_full
                time_shift = 0 if align_start else np.argmax(upsampled_mo_rate)
                # upsampled_mo_rate = upsampled_mo_rate / np.max(upsampled_mo_rate)
                if show_individual_mo:
                    plt.plot(max_time_array_in_bin_actual - time_shift, upsampled_mo_rate, alpha=.05,
                             color=cmap(norm(mw_all_events[idx])))
                binned_stack.append(upsampled_mo_rate)
                mo_list_stack.append(np.sum(mo_rate_list_all_events[idx]))

            # binned_stack = np.array(binned_stack)
            avg_moment_stack = np.mean(mo_list_stack)
            binned_stack = np.nansum(binned_stack, axis=0) / len(binned_stack)
            # binned_stack = binned_stack / np.max(binned_stack)
            '''if rescale_stack_mo:
                binned_stack = binned_stack * avg_moment_stack'''

            print('mo recomputed', mo_to_mw(np.sum(binned_stack)), mean_mw_per_bin)

            time_shift_stack = 0 if align_start else np.argmax(binned_stack)
            y_shift_stack = binned_stack[0] if rescale_zero_y else 0
            print('shift', y_shift_stack)
            # plt.plot(max_time_array - amax, binned_stack, color=colors[i], lw=2., label='weighted_stack')
            plt.plot(max_time_array_in_bin_actual - time_shift_stack, binned_stack - y_shift_stack,
                     color=colors[i], lw=2.)
            # Fit slip evolution function to stacked Mo rates
            opt_params = fit_slip_evolution(max_time_array_in_bin_actual, binned_stack, var_on_peak=1,
                                            slip_ev_fcn_signature=triangular_slip_evolution)
            A_fit, T0_fit, T1_fit, T2_fit = opt_params

            acc_times.append(T1_fit - T0_fit)
            mo_amplitudes.append(A_fit)
            avg_moments.append(avg_moment_stack)
            dec_times.append(T2_fit - T1_fit)

            slip_ev_fit = triangular_slip_evolution(max_time_array_in_bin_actual, *opt_params)
            if show_fit:
                plt.plot(max_time_array_in_bin_actual - time_shift_stack, slip_ev_fit - y_shift_stack, '--',
                         color=colors[i], lw=2.)
            # plt.show()
        fit_str = '_fit' if show_fit else ''
        n_bin_str = f'_{n_dur_bins}_bins'
        thresh_str = f'_thresh_{thresh}'

        cbar = plt.colorbar(sm, ax=axis)
        cbar.ax.set_ylabel('Moment magnitude (Mw)', rotation=270, labelpad=25)

        plt.ylabel('Moment rate function [$N\cdot m \cdot d^{-1}$]')

        if align_start:
            plt.xlabel('Duration [days]')
        else:
            plt.xlabel('Duration (relative to peak) [days]')
        plt.savefig(f'{local_subfolder}/eventwise_moment_rate_evolution{fit_str}{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        acc_times, mo_amplitudes, dec_times = np.array(acc_times), np.array(mo_amplitudes), np.array(dec_times)
        avg_moments = np.array(avg_moments)

        acc_times[acc_times > 100] = np.nan  # remove extreme values
        acc_times[acc_times < 0] = np.nan  # remove extreme values
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Acceleration time [days]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_acc_time_vs_duration{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, acc_times)

        '''def exponential_function(x, a, b, c):
            return a * np.exp(x * b) + c
        popt, pcov = curve_fit(exponential_function, avg_moments, acc_times, p0=(1, 1e-16, 1), nan_policy='omit')
        x_fit = np.linspace(np.nanmin(avg_moments), np.nanmax(avg_moments), 100)
        plt.plot(x_fit, exponential_function(x_fit, *popt), '--')'''

        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Acceleration time [days]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_acc_time_vs_mo{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)],
                    mo_amplitudes / acc_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Moment acceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_mo_acc_vs_duration{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, mo_amplitudes / acc_times)
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Moment acceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_mo_acc_vs_mo{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        # deceleration
        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], dec_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Deceleration time [days]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_dec_time_vs_duration{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, dec_times)
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Deceleration time [days]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_dec_time_vs_mo{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, (acc_times) / (acc_times + dec_times))
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Normalized acceleration time [%]')
        plt.ylim([0, 1])
        plt.savefig(f'{local_subfolder}/eventwise_fitted_accnorm_time_vs_mo{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)],
                    - mo_amplitudes / dec_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Moment deceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_mo_dec_vs_duration{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, - mo_amplitudes / dec_times)
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Moment deceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_mo_dec_vs_mo{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()


def mo_rate_stack_asymmetry_eventwise_mw_bins(sse_info_thresh, slip_thresholds, new_duration_dict, n_mw_bins=5,
                                              show_fit=False,
                                              show_individual_mo=True, align_start=False, rescale_zero_y=False,
                                              base_dir='figures', refine_durations=True):
    local_dir = os.path.join(base_dir, 'asymmetry')
    os.makedirs(local_dir, exist_ok=True)
    bins_folder = os.path.join(local_dir, f'{n_mw_bins}_bins_mw')
    os.makedirs(bins_folder, exist_ok=True)

    for thresh in slip_thresholds:
        thresh_folder = f'thresh_{thresh}'
        local_subfolder = os.path.join(bins_folder, thresh_folder)
        os.makedirs(local_subfolder, exist_ok=True)
        event_moment_list, event_duration_list, event_area_list, slip_event_list, patch_idx_list, date_list, mo_rate_list, slip_rate_list = \
            sse_info_thresh[thresh]

        if refine_durations:
            mo_rate_list_all_events = []
            for i, mr in enumerate(mo_rate_list):
                idx = new_duration_dict[thresh][i]
                # new_start = date_list[i][0] + idx[0]
                # new_end = date_list[i][0] + idx[1]
                new_start = idx[0]
                new_end = idx[1]
                mo_rate_list_all_events.append(np.sum(mr[new_start:new_end], axis=1))
        else:
            mo_rate_list_all_events = [np.sum(mo_rate_list[i], axis=1) for i in range(len(mo_rate_list))]

        if thresh == .07:
            print(len(mo_rate_list_all_events))
            for ev in mo_rate_list_all_events:
                plt.plot(ev)
            plt.show()

        mo_all_events = np.array([np.sum(mo_rate) for mo_rate in mo_rate_list_all_events])
        # remove events that are not valid
        valid_event_mask = mo_all_events > 0.
        mo_all_events = mo_all_events[valid_event_mask]
        mo_rate_list_all_events = [mo_rate_list_all_events[i] for i in range(len(mo_rate_list_all_events)) if
                                   valid_event_mask[i]]

        mw_all_events = np.array([mo_to_mw(mo) for mo in mo_all_events])  # can contain -infs

        # create bins with approximately the same number of samples
        mw_bins = np.percentile(mw_all_events[~np.isinf(mw_all_events)], np.linspace(0, 100, n_mw_bins + 1))

        # mw-related colorbar is computed based on mean Mo per bin
        mean_mo_per_bin = []
        for i in range(len(mw_bins) - 1):
            mw_bin = (mw_bins[i], mw_bins[i + 1])
            bin_idx = np.where((mw_all_events >= mw_bin[0]) & (mw_all_events < mw_bin[1]))[0]
            mean_mo_per_bin.append(np.mean([np.sum(mo_rate_list_all_events[j]) for j in bin_idx]))
            '''for j in bin_idx:
                plt.plot(mo_rate_list_all_events[j])
            plt.show()'''
        mean_mw_per_bin = [mo_to_mw(mo) for mo in mean_mo_per_bin]

        figure, axis = plt.subplots(1, 1, figsize=(10, 7), dpi=300, sharex='all', sharey='all')
        base_cmap = matplotlib.cm.Blues
        cmap = LinearSegmentedColormap.from_list(
            'Blues_darker', base_cmap(np.linspace(0.2, 1, 256)))
        colors = cmap(np.linspace(0, 1, len(mean_mw_per_bin)))
        norm = Normalize(vmin=np.nanmin(mean_mw_per_bin), vmax=np.nanmax(mean_mw_per_bin))
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        acc_times, dec_times, mo_amplitudes, avg_moments = [], [], [], []
        for i in range(len(mw_bins) - 1):
            mw_bin = (mw_bins[i], mw_bins[i + 1])
            print('bin:', mw_bin)
            bin_idx = np.where((mw_all_events >= mw_bin[0]) & (mw_all_events < mw_bin[1]))[0]
            print(len(bin_idx), 'events with Mw in', mw_bin)

            if len(bin_idx) == 0:  # or dur_bins[i] < 4.:  # maybe it could be done in a better way...
                acc_times.append(np.nan)  # add nan to skip a point in the plot (dur_bins is left untouched)
                dec_times.append(np.nan)
                mo_amplitudes.append(np.nan)
                avg_moments.append(np.nan)
                continue

            max_duration_bin = max([len(mo_rate_list_all_events[j]) for j in bin_idx])
            max_time_array_in_bin = np.linspace(0, 1, max_duration_bin)  # in [0,1] for correct resampling
            max_time_array_in_bin_actual = np.linspace(0, max_duration_bin,
                                                       max_duration_bin)  # to plot the right duration
            binned_stack, mo_list_stack = [], []
            for idx in bin_idx:
                mo_len = len(mo_rate_list_all_events[idx])
                if mo_len < 2:
                    continue
                print(f'try to upsample over {max_duration_bin} days:', len(mo_rate_list_all_events[idx]))
                f_int = interp1d(np.linspace(0, 1, mo_len), mo_rate_list_all_events[idx], kind='linear')
                upsampled_mo_rate = f_int(max_time_array_in_bin)

                n_smooth_kernel = 7
                if max_duration_bin > 10:  # perform smoothing on samples that are long enough
                    avg_smoothing_kernel = np.ones(n_smooth_kernel) / n_smooth_kernel
                    upsampled_mo_rate = np.convolve(upsampled_mo_rate, avg_smoothing_kernel, mode='same')

                # amax = np.argmax(upsampled_mo_rate) / max_duration_full
                time_shift = 0 if align_start else np.argmax(upsampled_mo_rate)
                # upsampled_mo_rate = upsampled_mo_rate / np.max(upsampled_mo_rate)
                if show_individual_mo:
                    plt.plot(max_time_array_in_bin_actual - time_shift, upsampled_mo_rate, alpha=.05,
                             color=cmap(norm(mw_all_events[idx])))
                binned_stack.append(upsampled_mo_rate)
                mo_list_stack.append(np.sum(mo_rate_list_all_events[idx]))

            # binned_stack = np.array(binned_stack)
            avg_moment_stack = np.mean(mo_list_stack)
            binned_stack = np.nansum(binned_stack, axis=0) / len(binned_stack)
            # binned_stack = binned_stack / np.max(binned_stack)
            '''if rescale_stack_mo:
                binned_stack = binned_stack * avg_moment_stack'''

            print('mo recomputed', mo_to_mw(np.sum(binned_stack)), mean_mw_per_bin)

            time_shift_stack = 0 if align_start else np.argmax(binned_stack)
            y_shift_stack = binned_stack[0] if rescale_zero_y else 0
            print('shift', y_shift_stack)
            # plt.plot(max_time_array - amax, binned_stack, color=colors[i], lw=2., label='weighted_stack')
            plt.plot(max_time_array_in_bin_actual - time_shift_stack, binned_stack - y_shift_stack,
                     color=colors[i], lw=2.)
            # Fit slip evolution function to stacked Mo rates
            '''opt_params = fit_slip_evolution(max_time_array_in_bin_actual, binned_stack, var_on_peak=1,
                                            slip_ev_fcn_signature=triangular_slip_evolution)
            A_fit, T0_fit, T1_fit, T2_fit = opt_params

            acc_times.append(T1_fit - T0_fit)
            mo_amplitudes.append(A_fit)
            avg_moments.append(avg_moment_stack)
            dec_times.append(T2_fit - T1_fit)

            slip_ev_fit = triangular_slip_evolution(max_time_array_in_bin_actual, *opt_params)
            if show_fit:
                plt.plot(max_time_array_in_bin_actual - time_shift_stack, slip_ev_fit - y_shift_stack, '--',
                         color=colors[i], lw=2.)'''
            # plt.show()
        fit_str = '_fit' if show_fit else ''
        n_bin_str = f'_{n_mw_bins}_mw_bins'
        thresh_str = f'_thresh_{thresh}'

        cbar = plt.colorbar(sm, ax=axis)
        cbar.ax.set_ylabel('Moment magnitude (Mw)', rotation=270, labelpad=25)

        plt.ylabel('Moment rate function [$N\cdot m \cdot d^{-1}$]')

        if align_start:
            plt.xlabel('Duration [days]')
        else:
            plt.xlabel('Duration (relative to peak) [days]')
        plt.savefig(f'{local_subfolder}/eventwise_moment_rate_evolution{fit_str}{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()
        continue
        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        acc_times, mo_amplitudes, dec_times = np.array(acc_times), np.array(mo_amplitudes), np.array(dec_times)
        avg_moments = np.array(avg_moments)

        acc_times[acc_times > 100] = np.nan  # remove extreme values
        acc_times[acc_times < 0] = np.nan  # remove extreme values
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Acceleration time [days]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_acc_time_vs_duration{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, acc_times)

        '''def exponential_function(x, a, b, c):
            return a * np.exp(x * b) + c
        popt, pcov = curve_fit(exponential_function, avg_moments, acc_times, p0=(1, 1e-16, 1), nan_policy='omit')
        x_fit = np.linspace(np.nanmin(avg_moments), np.nanmax(avg_moments), 100)
        plt.plot(x_fit, exponential_function(x_fit, *popt), '--')'''

        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Acceleration time [days]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_acc_time_vs_mo{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)],
                    mo_amplitudes / acc_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Moment acceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_mo_acc_vs_duration{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, mo_amplitudes / acc_times)
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Moment acceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_mo_acc_vs_mo{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        # deceleration
        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], dec_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Deceleration time [days]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_dec_time_vs_duration{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, dec_times)
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Deceleration time [days]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_dec_time_vs_mo{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, (acc_times) / (acc_times + dec_times))
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Normalized acceleration time [%]')
        plt.ylim([0, 1])
        plt.savefig(f'{local_subfolder}/eventwise_fitted_accnorm_time_vs_mo{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)],
                    - mo_amplitudes / dec_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Moment deceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_mo_dec_vs_duration{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, - mo_amplitudes / dec_times)
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Moment deceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_mo_dec_vs_mo{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()


def unified_spatiotemporal_mo_rate_analysis(se: SlowSlipEventExtractor, refined_durations, n_mw_bins=5, show_fit=False,
                                            show_individual_mo=True, align_start=False, rescale_zero_y=False,
                                            base_dir='figures'):
    for thresh in se.slip_thresholds:
        mo_rate_list_patches = se.get_moment_rate_events(thresh, refined_durations)
        mo_rate_list_all_events = [np.sum(mo_rate_list_patches[i], axis=1) for i in range(len(mo_rate_list_patches))]

        mo_all_events = np.array([np.sum(mo_rate) for mo_rate in mo_rate_list_all_events])
        # remove events that are not valid
        valid_event_mask = mo_all_events > 0.
        mo_all_events = mo_all_events[valid_event_mask]
        mo_rate_list_all_events = [mo_rate_list_all_events[i] for i in range(len(mo_rate_list_all_events)) if
                                   valid_event_mask[i]]

        mw_all_events = np.array([mo_to_mw(mo) for mo in mo_all_events])  # can contain -infs
        duration_all_events = np.array([len(mr) for mr in mo_rate_list_all_events])

        nucleation_idx, arrest_idx, valid_mask = se.get_start_end_patch(0.07, delta_win=5)
        nuc_x, nuc_y = se.ma.x_centr_km[nucleation_idx], se.ma.y_centr_km[nucleation_idx]
        arr_x, arr_y = se.ma.x_centr_km[arrest_idx], se.ma.y_centr_km[arrest_idx]
        valid_mw = se.get_magnitude_events(0.07, True)[valid_mask]

        distance = np.sqrt((nuc_x - arr_x) ** 2 + (nuc_y - arr_y) ** 2)

        print('average distance when Mw < 6.2:',
              np.mean(distance[(valid_mw < 6.2) & (valid_mw > 4.5) & (valid_mw != 0)]), 'km')
        avg_distance = np.mean(distance[(valid_mw < 6.2) & (valid_mw != 0)])

        # create bins with approximately the same number of samples
        mw_bins = np.percentile(mw_all_events[~np.isinf(mw_all_events)], np.linspace(0, 100, n_mw_bins + 1))

        # mw-related colorbar is computed based on mean Mo per bin
        mean_mo_per_bin = []
        for i in range(len(mw_bins) - 1):
            mw_bin = (mw_bins[i], mw_bins[i + 1])
            bin_idx = np.where((mw_all_events >= mw_bin[0]) & (mw_all_events < mw_bin[1]))[0]
            mean_mo_per_bin.append(np.mean([np.sum(mo_rate_list_all_events[j]) for j in bin_idx]))
            '''for j in bin_idx:
                plt.plot(mo_rate_list_all_events[j])
            plt.show()'''
        mean_mw_per_bin = [mo_to_mw(mo) for mo in mean_mo_per_bin]

        # figure, axis = plt.subplots(1, 1, figsize=(10, 7), dpi=300, sharex='all', sharey='all')
        # fig_scaling, ax_scaling = plt.subplots(1, 1, figsize=(10, 7), dpi=300, sharex='all', sharey='all')
        # fig_distance, ax_distance = plt.subplots(1, 1, figsize=(10, 7), dpi=300, sharex='all', sharey='all')

        figure, (ax_mo_rates, ax_distance, ax_scaling) = plt.subplots(1, 3, figsize=(28, 7), dpi=300)

        mwax = ax_scaling.secondary_xaxis('top', functions=(logmo_to_mw, mw_to_logmo))

        base_cmap = matplotlib.cm.Blues
        cmap = LinearSegmentedColormap.from_list(
            'Blues_darker', base_cmap(np.linspace(0.2, 1, 256)))
        colors = cmap(np.linspace(0, 1, len(mean_mw_per_bin)))
        norm = Normalize(vmin=np.nanmin(mean_mw_per_bin), vmax=np.nanmax(mean_mw_per_bin))
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        acc_times, dec_times, mo_amplitudes, avg_moments = [], [], [], []
        for i in range(len(mw_bins) - 1):
            mw_bin = (mw_bins[i], mw_bins[i + 1])
            print('bin:', mw_bin)
            bin_idx = np.where((mw_all_events >= mw_bin[0]) & (mw_all_events < mw_bin[1]))[0]
            print(len(bin_idx), 'events with Mw in', mw_bin)

            if len(bin_idx) == 0:  # or dur_bins[i] < 4.:  # maybe it could be done in a better way...
                acc_times.append(np.nan)  # add nan to skip a point in the plot (dur_bins is left untouched)
                dec_times.append(np.nan)
                mo_amplitudes.append(np.nan)
                avg_moments.append(np.nan)
                continue

            max_duration_bin = max([len(mo_rate_list_all_events[j]) for j in bin_idx])
            max_time_array_in_bin = np.linspace(0, 1, max_duration_bin)  # in [0,1] for correct resampling
            max_time_array_in_bin_actual = np.linspace(0, max_duration_bin,
                                                       max_duration_bin)  # to plot the right duration
            binned_stack, mo_list_stack = [], []
            for idx in bin_idx:
                mo_len = len(mo_rate_list_all_events[idx])
                if mo_len < 2:
                    continue
                print(f'try to upsample over {max_duration_bin} days:', len(mo_rate_list_all_events[idx]))
                print(np.linspace(0, 1, mo_len).shape, mo_rate_list_all_events[idx].shape)
                f_int = interp1d(np.linspace(0, 1, mo_len), mo_rate_list_all_events[idx], kind='linear')
                upsampled_mo_rate = f_int(max_time_array_in_bin)

                n_smooth_kernel = 7
                if max_duration_bin > 10:  # perform smoothing on samples that are long enough
                    avg_smoothing_kernel = np.ones(n_smooth_kernel) / n_smooth_kernel
                    upsampled_mo_rate = np.convolve(upsampled_mo_rate, avg_smoothing_kernel, mode='same')

                # amax = np.argmax(upsampled_mo_rate) / max_duration_full
                time_shift = 0 if align_start else np.argmax(upsampled_mo_rate)
                # upsampled_mo_rate = upsampled_mo_rate / np.max(upsampled_mo_rate)
                if show_individual_mo:
                    ax_mo_rates.plot(max_time_array_in_bin_actual - time_shift, upsampled_mo_rate, alpha=.05,
                                     color=cmap(norm(mw_all_events[idx])))
                binned_stack.append(upsampled_mo_rate)
                mo_list_stack.append(np.sum(mo_rate_list_all_events[idx]))

            # binned_stack = np.array(binned_stack)
            binned_stack = np.nansum(binned_stack, axis=0) / len(binned_stack)

            time_shift_stack = 0 if align_start else np.argmax(binned_stack)
            y_shift_stack = binned_stack[0] if rescale_zero_y else 0
            ax_mo_rates.plot(max_time_array_in_bin_actual - time_shift_stack, binned_stack - y_shift_stack,
                             color=colors[i], lw=2.)

            ax_scaling.scatter(np.log10(mo_all_events[bin_idx]), np.log10(duration_all_events[bin_idx] * 86400),
                               edgecolors=matplotlib.colors.colorConverter.to_rgba('black', alpha=.5),
                               c=colors[i])

            bin_idx_distance = np.where((valid_mw >= mw_bin[0]) & (valid_mw < mw_bin[1]))[0]
            ax_distance.scatter(valid_mw[bin_idx_distance], distance[bin_idx_distance], c=colors[i])

        '''mw_corner = 6.6
        mw_bounded_filter = mw_all_events > mw_corner
        result = linregress(np.log10(mo_all_events[mw_bounded_filter]), np.log10(duration_all_events[mw_bounded_filter] * 86400))
        bounded_slope, bounded_intercept = result.slope, result.intercept
        result = linregress(np.log10(mo_all_events[~mw_bounded_filter]), np.log10(duration_all_events[~mw_bounded_filter] * 86400))
        unbounded_slope, unbounded_intercept = result.slope, result.intercept
        print(1/bounded_slope, 1/unbounded_slope)

        ax_scaling.plot(np.log10(mo_all_events)[mw_bounded_filter],
                 np.log10(mo_all_events[mw_bounded_filter]) * bounded_slope + bounded_intercept)
        ax_scaling.plot(np.log10(mo_all_events)[~mw_bounded_filter],
                        np.log10(mo_all_events[~mw_bounded_filter]) * unbounded_slope + unbounded_intercept)'''

        ax_distance.hlines(y=avg_distance, xmin=4.8, xmax=6.2, colors='grey', alpha=.7, zorder=-1,
                           linestyles='dashed', linewidth=2)
        ax_distance.axvline(x=6.2, color='C0', alpha=.7, linestyle='dashed', zorder=-1)
        ax_distance.annotate('unbounded-bounded transition', xy=(6.1, 360), ha='center', va='center', rotation=90,
                             color='C0', fontsize=22.)
        ax_distance.annotate('$\sim$ 40 km', xy=(4.4, avg_distance), ha='center', va='center',
                             color='grey', fontsize=24.)
        ax_scaling.axvline(x=mw_to_logmo(6.2), color='C0', alpha=.7, linestyle='dashed', zorder=-1)

        # cbar = plt.colorbar(sm, ax=ax_mo_rates)
        cbar = plt.colorbar(sm, ax=ax_scaling)
        cbar.ax.set_ylabel('Moment magnitude (M$_w$)', rotation=270, labelpad=30)

        ax_mo_rates.set_ylabel('Moment rate function [$N\cdot m \cdot d^{-1}$]')

        if align_start:
            ax_mo_rates.set_xlabel('Duration [days]')
        else:
            ax_mo_rates.set_xlabel('Duration (relative to peak) [days]')

        ax_scaling.set_xlabel('log$_{10}$($M_0$)')
        ax_scaling.set_ylabel('log$_{10}$(duration (s))')
        mwax.set_xlabel('$M_w$')

        ax_distance.set_ylabel('Distance from nucleation to arrest point [km]')
        ax_distance.set_xlabel('Event M$_w$')

        '''figure.savefig(f'figures/figure4/fig_4_a_{thresh}.pdf', bbox_inches='tight')
        plt.close(figure)
        fig_scaling.savefig(f'figures/figure4/fig_4_c_{thresh}.pdf', bbox_inches='tight', transparent=True)
        plt.close(fig_scaling)
        fig_distance.savefig(f'figures/figure4/fig_4_b_{thresh}.pdf', bbox_inches='tight')
        plt.close(fig_distance)'''
        figure.savefig(f'figures/fig_4_{thresh}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        continue
        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        acc_times, mo_amplitudes, dec_times = np.array(acc_times), np.array(mo_amplitudes), np.array(dec_times)
        avg_moments = np.array(avg_moments)

        acc_times[acc_times > 100] = np.nan  # remove extreme values
        acc_times[acc_times < 0] = np.nan  # remove extreme values
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Acceleration time [days]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_acc_time_vs_duration{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, acc_times)

        '''def exponential_function(x, a, b, c):
            return a * np.exp(x * b) + c
        popt, pcov = curve_fit(exponential_function, avg_moments, acc_times, p0=(1, 1e-16, 1), nan_policy='omit')
        x_fit = np.linspace(np.nanmin(avg_moments), np.nanmax(avg_moments), 100)
        plt.plot(x_fit, exponential_function(x_fit, *popt), '--')'''

        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Acceleration time [days]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_acc_time_vs_mo{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)],
                    mo_amplitudes / acc_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Moment acceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_mo_acc_vs_duration{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, mo_amplitudes / acc_times)
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Moment acceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_mo_acc_vs_mo{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        # deceleration
        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], dec_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Deceleration time [days]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_dec_time_vs_duration{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, dec_times)
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Deceleration time [days]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_dec_time_vs_mo{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, (acc_times) / (acc_times + dec_times))
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Normalized acceleration time [%]')
        plt.ylim([0, 1])
        plt.savefig(f'{local_subfolder}/eventwise_fitted_accnorm_time_vs_mo{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)],
                    - mo_amplitudes / dec_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Moment deceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_mo_dec_vs_duration{n_bin_str}{thresh_str}.pdf',
                    bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, - mo_amplitudes / dec_times)
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Moment deceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_mo_dec_vs_mo{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()


def nucleation_arrest_point_vs_mw():
    se = SlowSlipEventExtractor()
    sse_info_thresh, new_duration_dict = se.get_extracted_events_unfiltered()
    nucleation_idx, arrest_idx, valid_mask = se.get_start_end_patch(0.07, delta_win=5)
    nuc_x, nuc_y = se.ma.x_centr_km[nucleation_idx], se.ma.y_centr_km[nucleation_idx]
    arr_x, arr_y = se.ma.x_centr_km[arrest_idx], se.ma.y_centr_km[arrest_idx]

    mw = se.get_magnitude_events(0.07, True)[valid_mask]
    mo_rates = se.get_moment_rate_events(0.07, True)

    d = np.sqrt((nuc_x - arr_x) ** 2 + (nuc_y - arr_y) ** 2)

    plt.scatter(mw, d)
    plt.ylabel('Distance from nucleation to arrest point [km]')
    plt.xlabel('Event M$_w$')
    plt.show()

    '''
    valid_mo_rates = [mo_rates[i] for i in range(len(mo_rates)) if valid_mask[i]]

    for i, mo_rate in enumerate(valid_mo_rates):
        if valid_mask[i]:
            for t in range(len(mo_rate)):
                sorted_indices = np.argsort(mo_rate[t])
                plt.scatter(t * np.ones(len(mo_rate[t])), se.ma.y_centr_lat[[sorted_indices]], c=mo_rate[t][[sorted_indices]])
            cbar = plt.colorbar()
            plt.scatter([0.], se.ma.y_centr_lat[nucleation_idx[i]], marker='x', s=50, color='red')
            plt.scatter([t], se.ma.y_centr_lat[arrest_idx[i]], marker='x', s=50, color='red')
            cbar.set_label('Moment Rate [N.m/day]')
            plt.ylabel('Latitude')
            plt.xlabel('Time [days]')
            plt.show()
    '''


def overview_latitude_time_plot(time, data, tremors, station_coordinates, latsort, offset=20, window_length=60,
                                static=False, downsample_tremors=False, draw_tremors=True, tremor_alpha=1.,
                                data_pcolormesh=False, zoom=False, show=True, dpi=200, trenchwards_only=True,
                                save_as='pdf', raster_scatter=True, modified_cmap=False):
    if trenchwards_only:
        vmin, vmax = 0, 0.1
        cmap = matplotlib.cm.get_cmap("turbo").copy()  # matplotlib.cm.get_cmap("turbo_r").copy()
        if modified_cmap:
            from matplotlib.colors import ListedColormap
            colors = cmap(np.linspace(0, 1, 256))
            light_grey = np.array([0.95, 0.95, 0.95, .5])
            n_blend = 30
            for i in range(n_blend):
                blend_factor = i / (n_blend - 1)  # 0 for i=0, 1 for i=n_blend-1
                colors[i] = (1 - blend_factor) * light_grey + blend_factor * colors[i]
            cmap = ListedColormap(colors)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    else:
        vmin, vmax = -0.02, 0.1
        cmap = matplotlib.cm.get_cmap("RdBu_r").copy()  # matplotlib.cm.get_cmap("turbo_r").copy()
        norm = TwoSlopeNorm(0., vmin=vmin, vmax=vmax)

    figure = plt.figure(figsize=(16, 9), dpi=dpi)
    if static:
        x, y = np.meshgrid(time[window_length // 2:-window_length // 2], station_coordinates[latsort, 0])
    else:
        if zoom:
            x, y = np.meshgrid(time, station_coordinates[latsort, 0])
        else:
            x, y = np.meshgrid(time[0 + offset:-window_length - offset], station_coordinates[latsort, 0])
    if data_pcolormesh:
        f = interp2d(x, y, data[:, latsort, 0].T, kind='cubic')
        x_up = np.linspace(time[window_length // 2], time[-window_length // 2], len(x) * 1)
        y_up = np.linspace(station_coordinates[latsort, 0][0], station_coordinates[latsort, 0][-1], len(y) * 1)
        data1 = f(x_up, y_up)
        Xn, Yn = np.meshgrid(x_up, y_up)
        plt.pcolormesh(Xn, Yn, data1, cmap=cmap, norm=norm, zorder=0)
        # plt.pcolormesh(x, y, data[:, latsort, 0].T,  cmap=cmap, norm=norm, zorder=0, antialiased=True, shading='gouraud')
    else:
        # plot in reverse order to avoid to mask the sse growth
        '''plt.scatter(x[:, ::-1], y[:, ::-1], c=data[:, latsort, 0].T[:, ::-1], cmap=cmap, norm=norm, s=10, alpha=0.7,
                    zorder=0, edgecolors='none')  # old and working'''
        max_sort_disp = np.argsort(-data[:, latsort, 0].T.flatten())
        sc = plt.scatter(x.flatten()[max_sort_disp], y.flatten()[max_sort_disp],
                         c=-data[:, latsort, 0].T.flatten()[max_sort_disp], cmap=cmap, norm=norm, s=5, alpha=0.7,
                         zorder=0, edgecolors='none', rasterized=raster_scatter)
        '''color = -data[:, latsort, 0].T.flatten()[max_sort_disp]
        color[:-100] = np.nan
        sc = plt.scatter(x.flatten()[max_sort_disp], y.flatten()[max_sort_disp],
                    c=color, cmap=cmap, norm=norm, s=10, alpha=0.7, zorder=0,
                    edgecolors='none')'''
        # plt.scatter(x, y, c=data[:, latsort, 0].T, cmap=cmap, norm=norm, s=10, alpha=0.5, zorder=0, edgecolors='none')
        # cbar = plt.colorbar(fraction=0.046, pad=0.02, shrink=0.9)

    if draw_tremors:
        tremor_scatter_size = 0.2
        if downsample_tremors:
            fraction_points_per_cluster = 0.05
            # we only downsample tremors for PNSN catalogue
            idx_pnsn = tremors[:, 3] > 2009
            # ide's catalogue is kept as it is
            plt.scatter(tremors[~idx_pnsn, 3], tremors[~idx_pnsn, 0], s=tremor_scatter_size, alpha=tremor_alpha,
                        color='black', zorder=1)
            dbscan = DBSCAN(eps=0.1, min_samples=10).fit(tremors[idx_pnsn][:, (0, 3)])
            labels = dbscan.labels_

            unique_labels = np.unique(labels)

            for label in unique_labels:
                cluster_points = np.where(labels == label)[0]
                n_points_per_cluster = int(fraction_points_per_cluster * len(cluster_points))
                # print('#points:', n_points_per_cluster)
                selected_indices = np.random.choice(cluster_points, size=n_points_per_cluster)
                plt.scatter(tremors[idx_pnsn][selected_indices, 3], tremors[idx_pnsn][selected_indices, 0],
                            s=tremor_scatter_size, alpha=tremor_alpha, color='black', zorder=1)
        else:
            plt.scatter(tremors[:, 3], tremors[:, 0], s=tremor_scatter_size, alpha=tremor_alpha, color='black',
                        zorder=1)
    plt.ylabel('Latitude [°]')
    plt.xlabel('Time [years]')
    # plt.xlim([2007, 2022.1])
    # plt.ylim([39.9, 50.5])
    plt.margins(x=0, y=0)

    ax_divider = make_axes_locatable(plt.gca())  # after definition of main labels
    cax = ax_divider.append_axes('top', size='5%', pad='3%')
    cbar = figure.colorbar(sc, cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    cbar.solids.set_alpha(1)
    # cbar.ax.set_ylabel('Denoised E-W displacement rate [mm/day]', rotation=270, labelpad=25)
    cbar.ax.set_xlabel('Denoised E-W displacement rate [mm/day]', labelpad=-60)
    '''if not trenchwards_only:
        # sc.set_clim([vmin, vmax])
        cbar.ax.minorticks_on()'''
    if show:
        plt.show()
    else:
        plt.savefig(f'figures/overview_disp_tremors.{save_as}', bbox_inches='tight')
        plt.close(figure)


def slip_latitude_time_plot(time, slip_rates, tremors, y_centr_lat, offset=20, window_length=60,
                            static=False, downsample_tremors=False, draw_tremors=True, tremor_alpha=1.,
                            data_pcolormesh=False, zoom=False, show=True, dpi=200, trenchwards_only=True,
                            save_as='pdf', raster_scatter=True, modified_cmap=False, n_lat_bins=100,
                            scatter_size=5):
    if trenchwards_only:
        vmin, vmax = 0, 0.5
        cmap = matplotlib.cm.get_cmap("turbo").copy()  # matplotlib.cm.get_cmap("turbo_r").copy()
        if modified_cmap:
            from matplotlib.colors import ListedColormap
            colors = cmap(np.linspace(0, 1, 256))
            light_grey = np.array([1, 1, 1, .5])
            n_blend = 50
            for i in range(n_blend):
                blend_factor = i / (n_blend - 1)  # 0 for i=0, 1 for i=n_blend-1
                colors[i] = (1 - blend_factor) * light_grey + blend_factor * colors[i]
            cmap = ListedColormap(colors)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    else:
        vmin, vmax = -0.02, 0.1
        cmap = matplotlib.cm.get_cmap("RdBu_r").copy()  # matplotlib.cm.get_cmap("turbo_r").copy()
        norm = TwoSlopeNorm(0., vmin=vmin, vmax=vmax)

    _, bin_edges = np.histogram(y_centr_lat, bins=n_lat_bins)
    lat_slip = np.zeros((len(time), len(bin_edges)))
    lat_slip.fill(np.nan)

    for i in range(len(bin_edges) - 1):
        idx_bin = np.where(np.logical_and(y_centr_lat >= bin_edges[i], y_centr_lat < bin_edges[i + 1]))[0]
        lat_slip[:, i] = np.mean(slip_rates[:, idx_bin], axis=1)

    x_mesh, y_mesh = np.meshgrid(time, bin_edges)
    max_sort_slip_idx = np.argsort(lat_slip.T.flatten())

    figure = plt.figure(figsize=(16, 9), dpi=dpi)

    if data_pcolormesh:  # delete soon
        f = interp2d(x, y, data[:, latsort, 0].T, kind='cubic')
        x_up = np.linspace(time[window_length // 2], time[-window_length // 2], len(x) * 1)
        y_up = np.linspace(station_coordinates[latsort, 0][0], station_coordinates[latsort, 0][-1], len(y) * 1)
        data1 = f(x_up, y_up)
        Xn, Yn = np.meshgrid(x_up, y_up)
        plt.pcolormesh(Xn, Yn, data1, cmap=cmap, norm=norm, zorder=0)
        # plt.pcolormesh(x, y, data[:, latsort, 0].T,  cmap=cmap, norm=norm, zorder=0, antialiased=True, shading='gouraud')
    else:
        # plot in reverse order to avoid to mask the sse growth

        sc = plt.scatter(x_mesh.flatten()[max_sort_slip_idx], y_mesh.flatten()[max_sort_slip_idx],
                         c=lat_slip.T.flatten()[max_sort_slip_idx], cmap=cmap, norm=norm, s=scatter_size, alpha=0.7,
                         zorder=0, edgecolors='none', rasterized=raster_scatter)
        '''color = -data[:, latsort, 0].T.flatten()[max_sort_disp]
        color[:-100] = np.nan
        sc = plt.scatter(x.flatten()[max_sort_disp], y.flatten()[max_sort_disp],
                    c=color, cmap=cmap, norm=norm, s=10, alpha=0.7, zorder=0,
                    edgecolors='none')'''
        # plt.scatter(x, y, c=data[:, latsort, 0].T, cmap=cmap, norm=norm, s=10, alpha=0.5, zorder=0, edgecolors='none')
        # cbar = plt.colorbar(fraction=0.046, pad=0.02, shrink=0.9)

    if draw_tremors:
        tremor_scatter_size = 0.2
        if downsample_tremors:
            fraction_points_per_cluster = 0.05
            # we only downsample tremors for PNSN catalogue
            idx_pnsn = tremors[:, 3] > 2009
            # ide's catalogue is kept as it is
            plt.scatter(tremors[~idx_pnsn, 3], tremors[~idx_pnsn, 0], s=tremor_scatter_size, alpha=tremor_alpha,
                        color='black', zorder=1)
            dbscan = DBSCAN(eps=0.1, min_samples=10).fit(tremors[idx_pnsn][:, (0, 3)])
            labels = dbscan.labels_

            unique_labels = np.unique(labels)

            for label in unique_labels:
                cluster_points = np.where(labels == label)[0]
                n_points_per_cluster = int(fraction_points_per_cluster * len(cluster_points))
                # print('#points:', n_points_per_cluster)
                selected_indices = np.random.choice(cluster_points, size=n_points_per_cluster)
                plt.scatter(tremors[idx_pnsn][selected_indices, 3], tremors[idx_pnsn][selected_indices, 0],
                            s=tremor_scatter_size, alpha=tremor_alpha, color='black', zorder=1)
        else:
            plt.scatter(tremors[:, 3], tremors[:, 0], s=tremor_scatter_size, alpha=tremor_alpha, color='black',
                        zorder=1)
    plt.ylabel('Latitude [°]')
    plt.xlabel('Time [years]')
    # plt.xlim([2007, 2022.1])
    # plt.ylim([39.9, 50.5])
    plt.margins(x=0, y=0)

    ax_divider = make_axes_locatable(plt.gca())  # after definition of main labels
    cax = ax_divider.append_axes('top', size='3%', pad='3%')
    cbar = figure.colorbar(sc, cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    cbar.solids.set_alpha(1)
    # cbar.ax.set_ylabel('Denoised E-W displacement rate [mm/day]', rotation=270, labelpad=25)
    cbar.ax.set_xlabel('Slip rates [mm/day]', labelpad=-60)
    '''if not trenchwards_only:
        # sc.set_clim([vmin, vmax])
        cbar.ax.minorticks_on()'''
    if show:
        plt.show()
    else:
        plt.savefig(f'figures/overview_slip_tremors.{save_as}', bbox_inches='tight')
        plt.close(figure)


def total_slip_subduction(slip, geometry, codes, coords):
    """Expects slip in meters."""
    slip = np.sum(slip, axis=0)

    figure, axis = plt.subplots(1, 1, figsize=(14, 7), dpi=300, sharex='all', sharey='all')
    cascadia_map = init_basemap_cascadia(axis)

    # cmap = custom_blues_colormap(teal=False, reversed=True)  # plt.cm.get_cmap('Blues')
    cmap = plt.cm.get_cmap('turbo')
    norm = Normalize(vmin=np.min(slip), vmax=np.max(slip))
    depth = - geometry[:, 11]
    for ii in range(len(geometry[:, 0])):
        if depth[ii] < 50 or True:  # exclude 0 values (dumping) (depth > 50km)
            x_geo, y_geo = UTM_GEO(geometry[ii, [12, 15, 18, 12]], geometry[ii, [13, 16, 19, 13]])
            x_fill, y_fill = UTM_GEO(geometry[ii, [12, 15, 18]], geometry[ii, [13, 16, 19]])
            x_fill, y_fill = cascadia_map(x_fill, y_fill)
            cascadia_map.plot(x_geo, y_geo, 'k', linewidth=0.2, latlon=True)  # PLOT THE TRIANGLES CONTOURS
            p, = axis.fill(x_fill, y_fill,
                           color=cmap(norm(slip[ii])))  # FILL THE TRIANGLES WITH THE DIP-STRIKE SLIP NORM

    station_scatter = cascadia_map.scatter(coords[:, 1], coords[:, 0], marker='^', s=15,
                                           color='C3', picker=True,
                                           latlon=True, edgecolors='black', linewidth=0.7)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Set an empty array to ensure the ScalarMappable has the correct norm
    # cbar = plt.colorbar(sm, ax=axis, orientation='vertical')
    # cbar.ax.set_ylabel('Cumulative slip [mm]', rotation=270, labelpad=20)
    # colorbar_axes = inset_axes(axis, width="30%", height="3%", loc='lower left')
    colorbar_axes = figure.add_axes([0.44, 0.18, 0.08, 0.015])  # [left, bottom, width, height]
    cbar = figure.colorbar(sm, cax=colorbar_axes, orientation='horizontal')
    cbar.ax.set_xlabel('Total slip [mm]', fontsize=12)

    levels = [20, 40, 60]
    admissible_depth, _ = read_depth_from_slab2(max_depth=100)
    x_dep, y_dep = admissible_depth[:, 0], admissible_depth[:, 1]
    depth = admissible_depth[:, 2]
    x_dep_map, y_dep_map = cascadia_map(x_dep, y_dep)
    isodepth = axis.tricontour(x_dep_map, y_dep_map, -depth, levels=levels, colors='k', linewidths=0.7)
    label_loc_y = [48.3, 41.7, 41.7]  # [48.3, 48.4]
    label_loc_x = [-125.5, -143.5, -122.0]  # [-125.5, -124.5]
    label_loc_x_map, label_loc_y_map = cascadia_map(label_loc_x, label_loc_y)

    clabel_txt = axis.clabel(isodepth, isodepth.levels, inline=True, fontsize=10, fmt=isodepth_label_fmt,
                             manual=list(zip(label_loc_x_map, label_loc_y_map)))
    # clabel_txt = axis.clabel(isodepth, isodepth.levels, inline=True, fontsize=12, fmt=isodepth_label_fmt)

    idx_albh = codes.index('ALBH')
    cascadia_map.scatter(coords[idx_albh, 1], coords[idx_albh, 0], latlon=True, color='C2', edgecolors='k', marker='^', s=40, zorder=10)

    '''bbox={'boxstyle': 'square,pad=0.3', 'facecolor': 'white', 'edgecolor': 'none'}
    for txt in clabel_txt:
        txt.set_bbox(bbox)'''

    plt.savefig('figures/tot_slip.pdf', bbox_inches='tight')
    plt.close(figure)
    # plt.show()


def albh_figure(time, data, modeled_ts, station_codes, albh_zip_data, offset_cut, station_code='ALBH', window_length=60,
                dpi=300):
    # fig = plt.figure(figsize=(21, 7), dpi=300)
    figure, axis = plt.subplots(1, 1, figsize=(21, 7), dpi=dpi, sharex='all', sharey='all')

    albh_index = station_codes.index(station_code) if type(station_codes) == list else station_codes.tolist().index(
        station_code)
    albh_cumsum = np.cumsum(data[:, albh_index, 0])
    model_albh_cumsum = np.cumsum(modeled_ts[:, albh_index, 0])

    markers, caps, bars = axis.errorbar(time, detrend(albh_zip_data[0]), yerr=albh_zip_data[1], fmt='o', capsize=2.5,
                                        capthick=.5, ecolor='grey', elinewidth=1., color='steelblue', alpha=.6)
    [bar.set_alpha(0.35) for bar in bars]
    [cap.set_alpha(0.35) for cap in caps]

    axis.plot(time[offset_cut:-(window_length + offset_cut)], detrend(albh_cumsum), linewidth=2., color='crimson',
              label='Denoised', zorder=10)
    axis.plot(time[offset_cut:-(window_length + offset_cut)], detrend(model_albh_cumsum), linewidth=2.,
              color='midnightblue', label='Model', zorder=10)
    # model is missing
    plt.xlabel('Time [years]')
    plt.ylabel('Displacement [mm]')
    # plt.xlim([2013.45, 2016.35])
    plt.xlim([2015.15, 2017.75])
    plt.ylim([-8, 8])
    plt.legend()
    plt.savefig('figures/albh_denoised_model.pdf', bbox_inches='tight')
    plt.close(figure)


def moment_rate_figure_comparison(se: SlowSlipEventExtractor, time_array, dpi=300, refine_durations=True):
    def format_sci(num):
        base, exp = f"{num:.2e}".split("e")
        exp = int(exp)
        return f"{base}$\\times $10$^{{{exp}}}$"

    # thresh 0.07, ev id (starting from 0) -> 42 (2015), 47 (2016)
    # thresh 0.3, ev id (starting from 0) -> 48 (2015), 50-51-52 (2016)
    # thresh 0.5, ev id (starting from 0) -> 23 (2015), 24-25-26 (2016)
    # thresh 0.7, ev id (starting from 0) -> 8 (2015), 9-10 (2016)
    id_events_2015 = {0.07: [42], 0.3: [48], 0.5: [23], 0.7: [8]}
    id_events_2016 = {0.07: [47], 0.3: [50, 51, 52], 0.5: [24, 25, 26], 0.7: [9, 10]}
    id_events_2017 = {0.07: [55, 56], 0.3: [53, 54], 0.5: [27, 28], 0.7: [11]}
    colors_thresh = {0.07: 'C0', 0.3: 'C1', 0.5: 'C2', 0.7: 'C3'}

    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(21, 7), dpi=dpi)
    # 2015 event
    for thresh in se.slip_thresholds:
        event_moment_rate_list = se.get_moment_rate_events(thresh, refine_durations)
        event_duration_list = np.array([len(mo_rate) for mo_rate in event_moment_rate_list])
        event_date_idx_list = se.get_event_date_idx(thresh, refine_durations)
        '''for id_event in id_events_2015[thresh]:
            mo_rate = np.sum(event_moment_rate_list[id_event], axis=1)
            ev_start_idx, ev_end_idx = event_date_idx_list[id_event]
            date_array = time_array[ev_start_idx: ev_end_idx + 1]
            axis1.plot(date_array, mo_rate, linewidth=2, color=colors_thresh[thresh], label=f'thresh: {thresh} mm/day')'''
        axis1.plot([], [], linewidth=2, color=colors_thresh[thresh])
        tot_mo = 0
        for i, id_event in enumerate(id_events_2016[thresh]):
            mo_rate = np.sum(event_moment_rate_list[id_event], axis=1)
            ev_start_idx, ev_end_idx = event_date_idx_list[id_event]
            date_array = time_array[ev_start_idx: ev_end_idx + 1]
            # axis2.plot(date_array, mo_rate, linewidth=2, color=colors_thresh[thresh])
            axis1.plot(date_array, mo_rate, linewidth=2, color=colors_thresh[thresh])
            tot_mo += np.sum(mo_rate)
            axis1.fill_between(date_array, mo_rate, color=colors_thresh[thresh], alpha=0.4,
                               label=f'$M_0=${format_sci(tot_mo)}$ N\cdot m$' if i == len(
                                   id_events_2016[thresh]) - 1 else None)
            print(f'1----$M_0=${format_sci(np.sum(mo_rate))}$ N\cdot m$')
        tot_mo = 0
        for j, id_event in enumerate(id_events_2017[thresh]):
            mo_rate = np.sum(event_moment_rate_list[id_event], axis=1)
            ev_start_idx, ev_end_idx = event_date_idx_list[id_event]
            date_array = time_array[ev_start_idx: ev_end_idx + 1]
            axis2.plot(date_array, mo_rate, linewidth=2, color=colors_thresh[thresh])
            tot_mo += np.sum(mo_rate)
            axis2.fill_between(date_array, mo_rate, color=colors_thresh[thresh], alpha=0.4,
                               label=f'$M_0=${format_sci(tot_mo)}$ N\cdot m$' if j == len(
                                   id_events_2017[thresh]) - 1 else None)
            print(f'2----$M_0=${format_sci(np.sum(mo_rate))}$ N\cdot m$')
        legend1 = axis1.legend(prop={'size': 22})
        legend2 = axis2.legend(prop={'size': 22})
        axis1.ticklabel_format(useOffset=False)
        axis2.ticklabel_format(useOffset=False)
    for ax in (axis1, axis2):
        ax.set_ylabel('Moment rate relative to\nthreshold [$N\cdot m \cdot d^{-1}$]')
        ax.set_xlabel('Time [years]')
    axis1.xaxis.set_major_locator(plt.MaxNLocator(3))

    plt.tight_layout()
    plt.savefig('figures/mo_rate_comparison.pdf', bbox_inches='tight')
    plt.close(fig)


def corner_freq_analysis(se: SlowSlipEventExtractor, refine_durations: bool, dpi: int = 100):
    slip_thresh = 0.07
    event_moment_rate_list = se.get_moment_rate_events(slip_thresh, refine_durations)
    event_moment_list = np.array([mw_to_mo(mw) for mw in se.get_magnitude_events(slip_thresh, refine_durations)])
    event_duration_list = np.array([len(mo_rate) for mo_rate in event_moment_rate_list])

    valid_mo_idx = np.where(event_moment_list > 0.)[0]
    event_moment_list = event_moment_list[valid_mo_idx]
    event_duration_list = event_duration_list[valid_mo_idx]
    event_mw_list = np.array([mo_to_mw(mo) for mo in event_moment_list])

    corner_freq = 1 / event_duration_list
    fig, axis = plt.subplots(1, 1, dpi=dpi)
    mwax = axis.secondary_xaxis('top', functions=(logmo_to_mw, mw_to_logmo))

    mw_corner = 6.2
    mw_bounded_filter = event_mw_list > mw_corner

    plt.scatter(np.log10(event_moment_list)[mw_bounded_filter], np.log10(corner_freq)[mw_bounded_filter])
    plt.scatter(np.log10(event_moment_list)[~mw_bounded_filter], np.log10(corner_freq)[~mw_bounded_filter])

    result = linregress(np.log10(event_moment_list[mw_bounded_filter]), np.log10(corner_freq[mw_bounded_filter]))
    bounded_slope, bounded_intercept = result.slope, result.intercept
    result = linregress(np.log10(event_moment_list[~mw_bounded_filter]), np.log10(corner_freq[~mw_bounded_filter]))
    unbounded_slope, unbounded_intercept = result.slope, result.intercept

    ransac = RANSACRegressor()
    ransac.fit(np.log10(event_moment_list[mw_bounded_filter]).reshape(-1, 1),
               np.log10(corner_freq[mw_bounded_filter]).reshape(-1, 1))
    yransac = ransac.predict(np.log10(event_moment_list[mw_bounded_filter]).reshape(-1, 1))
    bounded_ransac_coef = ransac.estimator_.coef_
    print('bounded_ransac_coef for fc', 1 / bounded_ransac_coef)

    plt.plot(np.log10(event_moment_list)[mw_bounded_filter],
             np.log10(event_moment_list)[mw_bounded_filter] * unbounded_slope + unbounded_intercept)
    plt.plot(np.log10(event_moment_list)[~mw_bounded_filter],
             np.log10(event_moment_list)[~mw_bounded_filter] * unbounded_slope + unbounded_intercept)
    # plt.plot(np.log10(event_moment_list)[mw_bounded_filter], yransac.flatten())
    plt.plot(np.log10(event_moment_list)[mw_bounded_filter],
             np.log10(event_moment_list)[mw_bounded_filter] * (-1) + unbounded_intercept + 13.2)

    print('unbounded_slope for fc', 1 / unbounded_slope)
    print('bounded_slope for fc', 1 / bounded_slope)

    plt.ylabel('log[corner frequency]')
    plt.xlabel('log[Mo]')
    mwax.set_xlabel('Mw')
    plt.show()


def total_moment_release(time, mo_rates):
    figure, axis = plt.subplots(1, 1, dpi=300, figsize=(24, 4))
    axis.plot(time, mo_rates, linewidth=2)
    axis.set_xlabel('Time [years]')
    axis.set_ylabel('Moment rate [$N \cdot m \cdot s^{-1}$]')
    axis.margins(x=0)
    plt.savefig('figures/total_moment_release.pdf', bbox_inches='tight')
    plt.close(figure)
