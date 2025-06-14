import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import interp1d

from config_files.plotting_style import set_matplotlib_style, get_style_dict, temporary_matplotlib_syle
from sse_extraction.SlowSlipEventExtractor import SlowSlipEventExtractor
from utils.geo_functions import mo_to_mw

set_matplotlib_style()

if __name__ == '__main__':
    current_style_dict = get_style_dict()

    n_mw_bins = 7
    align_start = True
    show_individual_mo = True
    rescale_zero_y = False
    refine_durations = True
    se = SlowSlipEventExtractor()
    se.load_extracted_events_unfiltered()
    time_array = se.ddh.corrected_time
    thresh = se.slip_thresholds[0]

    mo_rate_list_patches = se.get_moment_rate_events(thresh, refine_durations)
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

    mw_bins = np.percentile(mw_all_events[~np.isinf(mw_all_events)], np.linspace(0, 100, n_mw_bins + 1))

    # mw-related colorbar is computed based on mean Mo per bin
    mean_mo_per_bin = []
    mo_rates_per_bin = []
    for i in range(len(mw_bins) - 1):
        mw_bin = (mw_bins[i], mw_bins[i + 1])
        bin_idx = np.where((mw_all_events >= mw_bin[0]) & (mw_all_events < mw_bin[1]))[0]
        mean_mo_per_bin.append(np.mean([np.sum(mo_rate_list_all_events[j]) for j in bin_idx]))
        mo_rates_per_bin.append([mo_rate_list_all_events[j] for j in bin_idx])

    mean_mw_per_bin = [mo_to_mw(mo) for mo in mean_mo_per_bin]

    temporary_style = current_style_dict.copy()
    temporary_style['axes.titlesize'] += 4.
    temporary_style['axes.labelsize'] += 4.
    temporary_style['xtick.labelsize'] += 4.
    with temporary_matplotlib_syle(temporary_style):

        figure, (axes) = plt.subplots(2, 4, figsize=(23, 11), dpi=100, sharey=False)
        ax_mo_rates, *axes_bin = axes.ravel()
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
                    axes_bin[i].plot(max_time_array_in_bin_actual - time_shift, upsampled_mo_rate, alpha=.5,
                                     color=cmap(norm(mw_all_events[idx])))
                    axes_bin[i].set_title(f'$M_w \in${round(mw_bins[i], 2), round(mw_bins[i + 1], 2)}')
                    if i >= 3:
                        axes_bin[i].set_xlabel('Duration [days]')
                binned_stack.append(upsampled_mo_rate)
                mo_list_stack.append(np.sum(mo_rate_list_all_events[idx]))

            # binned_stack = np.array(binned_stack)
            binned_stack = np.nansum(binned_stack, axis=0) / len(binned_stack)

            time_shift_stack = 0 if align_start else np.argmax(binned_stack)
            y_shift_stack = binned_stack[0] if rescale_zero_y else 0
            ax_mo_rates.plot(max_time_array_in_bin_actual - time_shift_stack, binned_stack - y_shift_stack,
                             color=colors[i], lw=2.)
            axes_bin[i].plot(max_time_array_in_bin_actual - time_shift_stack, binned_stack - y_shift_stack,
                             color='k', lw=2.)

            bin_idx_distance = np.where((valid_mw >= mw_bin[0]) & (valid_mw < mw_bin[1]))[0]

        plt.subplots_adjust(hspace=0.3)

        # cbar = plt.colorbar(sm, ax=ax_mo_rates)
        cbar = plt.colorbar(sm, ax=axes.ravel().tolist())
        cbar.ax.set_ylabel('Moment magnitude (M$_w$)', rotation=270, labelpad=30)

        ax_mo_rates.set_ylabel('Moment rate function [$N\cdot m \cdot d^{-1}$]')

        ax_mo_rates.set_title('All stacks')

        axes_bin[3].set_ylabel('Moment rate function [$N\cdot m \cdot d^{-1}$]')
        '''if align_start:
            ax_mo_rates.set_xlabel('Duration [days]')
        else:
            ax_mo_rates.set_xlabel('Duration (relative to peak) [days]')'''

        plt.savefig('figures/mo_rates_stack_all.pdf', bbox_inches='tight')
