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

from FUNCTIONS.functions_slab import UTM_GEO
from utils.colormaps import custom_blues_colormap
from utils.fourier_transform import custom_fft
from utils.geo_functions import mo_to_mw
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
                                num_freq_bins=10):
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
            fft_event_list.append(fft_event)
            duration_list.append(len(filtered_mo_rate_event))
            freq_list.append(freq_event)

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

        global_spectrum_y_offset = 10 ** 0.1
        plt.plot(freq[freq_mask],
                 10 ** (global_slope * np.log10(freq[freq_mask]) + global_intercept + global_spectrum_y_offset), '--',
                 lw=2., color='k')

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
        plt.ylabel('Normalized $\dot M_0$ power spectrum $|\widehat \dot M_0(f)|^2$ / $M_0^2$')

        label_freq_pos = freq[np.abs(freq - 1 / 20) < 0.001][0]  # center the text at about 1/20 days^-1
        _annotate_line(axis, label_freq_pos, global_slope, global_intercept + global_spectrum_y_offset,
                       f'$\propto f^{{-n}}, n={round(abs(global_slope), 2)}$')


        plt.savefig(f'figures/mo_rate_spectra/mo_rate_spectrum_{thresh}.pdf', bbox_inches='tight')
        # plt.show()
        plt.close(figure)


def mo_rate_stack_asymmetry_patchwise(sse_info_thresh, slip_thresholds, n_dur_bins=5, show_fit=False, show_individual_mo=True,
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
                #upsampled_mo_rate = upsampled_mo_rate / np.max(upsampled_mo_rate)
                if show_individual_mo:
                    plt.plot(max_time_array_in_bin_actual - time_shift, upsampled_mo_rate, alpha=.05,
                             color=cmap(norm(mw_all_patches[idx])))
                binned_stack.append(upsampled_mo_rate)
                mo_list_stack.append(np.sum(mo_rate_list_all_patches[idx]))

            # binned_stack = np.array(binned_stack)
            avg_moment_stack = np.mean(mo_list_stack)
            binned_stack = np.nansum(binned_stack, axis=0) / len(binned_stack)
            #binned_stack = binned_stack / np.max(binned_stack)
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
            #plt.show()
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
                #new_start = date_list[i][0] + idx[0]
                #new_end = date_list[i][0] + idx[1]
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
                #upsampled_mo_rate = upsampled_mo_rate / np.max(upsampled_mo_rate)
                if show_individual_mo:
                    plt.plot(max_time_array_in_bin_actual - time_shift, upsampled_mo_rate, alpha=.05,
                             color=cmap(norm(mw_all_events[idx])))
                binned_stack.append(upsampled_mo_rate)
                mo_list_stack.append(np.sum(mo_rate_list_all_events[idx]))

            # binned_stack = np.array(binned_stack)
            avg_moment_stack = np.mean(mo_list_stack)
            binned_stack = np.nansum(binned_stack, axis=0) / len(binned_stack)
            #binned_stack = binned_stack / np.max(binned_stack)
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
            #plt.show()
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
        plt.savefig(f'{local_subfolder}/eventwise_fitted_acc_time_vs_duration{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
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
        plt.savefig(f'{local_subfolder}/eventwise_fitted_acc_time_vs_mo{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], mo_amplitudes / acc_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Moment acceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_mo_acc_vs_duration{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
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
        plt.savefig(f'{local_subfolder}/eventwise_fitted_dec_time_vs_duration{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, dec_times)
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Deceleration time [days]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_dec_time_vs_mo{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)
        # plt.show()

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter(avg_moments, (acc_times) / (acc_times + dec_times))
        plt.xlabel('Total moment [$N\cdot m$]')
        plt.ylabel('Normalized acceleration time [%]')
        plt.ylim([0, 1])
        plt.savefig(f'{local_subfolder}/eventwise_fitted_accnorm_time_vs_mo{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
        plt.close(figure)

        figure, axis = plt.subplots(1, 1, figsize=(8, 6), dpi=300, sharex='all', sharey='all')
        # plt.plot([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], acc_times * 100)
        plt.scatter([0.5 * (dur_bins[i] + dur_bins[i + 1]) for i in range(len(dur_bins) - 1)], - mo_amplitudes / dec_times)
        plt.xlabel('Duration [days]')
        plt.ylabel('Moment deceleration [$N\cdot m \cdot d^{-2}$]')
        plt.savefig(f'{local_subfolder}/eventwise_fitted_mo_dec_vs_duration{n_bin_str}{thresh_str}.pdf', bbox_inches='tight')
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


def overview_latitude_time_plot(time, data, tremors, station_coordinates, latsort, offset=20, window_length=60,
                                static=False, downsample_tremors=False, draw_tremors=True, tremor_alpha=1.,
                                data_pcolormesh=False, zoom=False, show=True, dpi=200, trenchwards_only=True,
                                save_as='pdf', raster_scatter=True):
    if trenchwards_only:
        vmin, vmax = 0, 0.1
        cmap = matplotlib.cm.get_cmap("turbo").copy()  # matplotlib.cm.get_cmap("turbo_r").copy()
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


def total_slip_subduction(slip, geometry, coords):
    """Expects slip in meters."""
    slip = np.sum(slip, axis=0)

    figure, axis = plt.subplots(1, 1, figsize=(14, 7), dpi=300, sharex='all', sharey='all')
    cascadia_map = init_basemap_cascadia(axis)

    cmap = custom_blues_colormap(teal=False, reversed=True)  # plt.cm.get_cmap('Blues')
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

    clabel_txt = axis.clabel(isodepth, isodepth.levels, inline=True, fontsize=10, fmt=isodepth_label_fmt, manual=list(zip(label_loc_x_map, label_loc_y_map)))
    #clabel_txt = axis.clabel(isodepth, isodepth.levels, inline=True, fontsize=12, fmt=isodepth_label_fmt)

    '''bbox={'boxstyle': 'square,pad=0.3', 'facecolor': 'white', 'edgecolor': 'none'}
    for txt in clabel_txt:
        txt.set_bbox(bbox)'''

    plt.savefig('figures/tot_slip.pdf', bbox_inches='tight')
    plt.close(figure)
    # plt.show()


def albh_figure(time, data, modeled_ts, station_codes, albh_zip_data, offset_cut, window_length=60):
    # fig = plt.figure(figsize=(21, 7), dpi=300)
    figure, axis = plt.subplots(1, 1, figsize=(21, 7), dpi=300, sharex='all', sharey='all')

    albh_index = station_codes.index('ALBH') if type(station_codes) == list else station_codes.tolist().index('ALBH')
    albh_cumsum = np.cumsum(data[:, albh_index, 0])
    model_albh_cumsum = np.cumsum(modeled_ts[:, albh_index, 0])

    markers, caps, bars = axis.errorbar(time, detrend(albh_zip_data[0]), yerr=albh_zip_data[1], fmt='o', capsize=2.5, capthick=.5, ecolor='grey', elinewidth=1., color='steelblue', alpha=.6)
    [bar.set_alpha(0.35) for bar in bars]
    [cap.set_alpha(0.35) for cap in caps]

    axis.plot(time[offset_cut:-(window_length + offset_cut)], detrend(albh_cumsum), linewidth=2., color='crimson', label='Denoised', zorder=10)
    axis.plot(time[offset_cut:-(window_length + offset_cut)], detrend(model_albh_cumsum), linewidth=2., color='midnightblue', label='Model', zorder=10)
    # model is missing
    plt.xlabel('Time [years]')
    plt.ylabel('Displacement [mm]')
    plt.xlim([2013.45, 2016.35])
    plt.ylim([-8, 8])
    plt.legend()
    plt.savefig('figures/albh_denoised_model.pdf', bbox_inches='tight')
    plt.close(figure)
