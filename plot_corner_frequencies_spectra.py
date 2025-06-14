import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import linregress
from statsmodels.regression.quantile_regression import QuantReg

from sse_extraction.SlowSlipEventExtractor import SlowSlipEventExtractor
from utils.fourier_transform import custom_fft
from utils.geo_functions import mo_to_mw

if __name__ == '__main__':
    sf = 1
    num_freq_bins = 40
    refine_durations = True
    se = SlowSlipEventExtractor()
    se.load_extracted_events_unfiltered()
    time_array = se.ddh.corrected_time
    thresh = se.slip_thresholds[0]
    slip = se.ma.slip_rates
    area = se.ma.area
    figure, axis = plt.subplots(1, 1, figsize=(8, 7), dpi=100, sharex='all', sharey='all')

    mo_rate = np.sum(se.ma.mo_rates, axis=1)  # total moment release
    freq, fft = custom_fft(mo_rate, sf, len(time_array), axis=0, pos_freq_only=True)

    tot_mo_rate = np.sum(mo_rate)
    fft = fft ** 2 / tot_mo_rate ** 2  # we look at the power, as Hawthorne and Bartlow (2018)

    plt.plot(freq, fft, color='k', lw=2.)

    min_global_freq = 1 / 100  # day^-1, minimum frequency for global slope calculation
    freq_mask = freq > min_global_freq

    result = linregress(np.log10(freq[freq_mask]), np.log10(fft[freq_mask]))
    global_slope, global_intercept = result.slope, result.intercept


    mo_rate_list = se.get_moment_rate_events(thresh, refine_durations)
    date_list = se.get_event_date_idx(thresh, refine_durations)
    event_moment_list = [np.sum(mr) for mr in mo_rate_list]
    event_duration_list = np.array([len(mr) for mr in se.get_moment_rate_events(thresh, refine_durations)])
    event_moment_list, event_duration_list = np.array(event_moment_list), np.array(event_duration_list)
    patch_idx_list = se.get_event_patches(thresh, refine_durations)

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
        if start_idx - ev_dur < 0 or end_idx + ev_dur > len(mo_rate):
            continue
        ext_slip_potency = slip_potency[start_idx - 0:end_idx + 1 + 0, patch_idx_list[i]].copy()
        mo_rate_event = ext_slip_potency * se.ma.shear_modulus

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

        if mw_list[i] >= 6.2:
            plt.plot(freq_event[0], fft_event[0], 'o', color='black', zorder=1000)
        else:
            if mw_list[i] > 5.2:
                plt.scatter(freq_event[0], fft_event[0], marker='x', color='black', zorder=1000, linewidths=1.5)
            else:
                plt.scatter(freq_event[0], fft_event[0], marker='x', color='grey', zorder=1000, linewidths=1.5)
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


    # perform linear fitting of corner frequencies, suppose a knee at about Mw 6.2 to have two separate regressions
    mw_thresh_bounded = 6.2
    # we take the first value of the fft, assuming that f_c ~ 1/T
    corner_frequencies = np.array([frq[0] for frq in freq_list])
    if thresh == 0.07:
        y_fc = np.array([ft[0] for ft in fft_event_list])  # fft values evaluated at corner freq
        mw_filter_bounded = np.array(valid_mw_list) > mw_thresh_bounded

        mw_filter_unbounded = np.logical_and(np.array(valid_mw_list) > 5.2, np.array(valid_mw_list) < mw_thresh_bounded)
        result = linregress(np.log10(corner_frequencies[mw_filter_unbounded]), np.log10(y_fc[mw_filter_unbounded]))
        unbounded_slope, unbounded_intercept = result.slope, result.intercept
        result = linregress(np.log10(corner_frequencies[mw_filter_bounded]), np.log10(y_fc[mw_filter_bounded]))
        bounded_slope, bounded_intercept = result.slope, result.intercept
        print('bounded slope', bounded_slope)
        print('unbounded slope', unbounded_slope)
        print('unbounded intercept', unbounded_intercept)
        print('bounded intercept', bounded_intercept)

        plt.plot(corner_frequencies[~mw_filter_bounded],
                 10 ** (unbounded_slope * np.log10(corner_frequencies[~mw_filter_bounded]) + unbounded_intercept - .6),
                 '--',
                 color='black', zorder=1000)
        plt.plot(corner_frequencies[mw_filter_bounded],
                 10 ** (bounded_slope * np.log10(corner_frequencies[mw_filter_bounded]) + bounded_intercept), '--',
                 color='black', zorder=1000)
        excluded_ev = np.where(np.array(valid_mw_list) < 5.2)[0]

    # print('unbounded_slope', unbounded_slope)
    # print('bounded_slope', bounded_slope)
    plt.savefig('figures/corner_freq_spectra.pdf', bbox_inches='tight')
