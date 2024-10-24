import numpy as np
from matplotlib import pyplot as plt

from config_files.plotting_style import set_matplotlib_style
from denoised_data_utils.DenoisedDataHandler import DenoisedDataHandler
from sse_extraction.SlowSlipEventExtractor import SlowSlipEventExtractor

set_matplotlib_style()

if __name__ == '__main__':

    se = SlowSlipEventExtractor()
    sse_info_thresh, new_duration_dict = se.get_extracted_events_unfiltered()
    nucleation_idx, arrest_idx, valid_mask = se.get_start_end_patch(0.07, delta_win=5)
    nuc_x, nuc_y = se.ma.x_centr_km[nucleation_idx], se.ma.y_centr_km[nucleation_idx]
    print(nuc_x > 1358)
    arr_x, arr_y = se.ma.x_centr_km[arrest_idx], se.ma.y_centr_km[arrest_idx]

    mw = se.get_magnitude_events(0.07, True)[valid_mask]
    mo_rates = se.get_moment_rate_events(0.07, True)

    d = np.sqrt((nuc_x - arr_x)**2 + (nuc_y - arr_y)**2)

    plt.scatter(mw, d)
    plt.ylabel('Distance from nucleation to arrest point [km]')
    plt.xlabel('Event M$_w$')
    plt.show()

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
































    exit(0)
    for thresh in se.slip_thresholds:
        event_moment_list, dl, event_area_list, slip_event_list, patch_idx_list, date_list, mo_rate_list, slip_rate_list = \
            sse_info_thresh[thresh]
        for i, mr in enumerate(mo_rate_list):
            idx = new_duration_dict[thresh][i]
            #print('original date', date_list[i])
            #print('new date idx', idx)
            new_start = date_list[i][0] + idx[0]
            new_end = date_list[i][0] + idx[1]

    '''print(se.ma.area)
    plt.hist(se.ma.area * 1e-6, bins=20)
    plt.xlabel('Area [km$^2$]')
    plt.show()'''

    # calcul du stress drop
    km2tom2 = 1e06
    sd,mos = [],[]
    for thresh in se.slip_thresholds:
        event_moment_list, dl, event_area_list, slip_event_list, patch_idx_list, date_list, mo_rate_list, slip_rate_list = \
            sse_info_thresh[thresh]
        '''for i, mr in enumerate(mo_rate_list):
            idx = new_duration_dict[thresh][i]
            new_start = idx[0]
            new_end = idx[1]
            mo = np.sum(mr[new_start:new_end])
            stressdrop = np.sum(event_area_list[i] * km2tom2) ** (- 3/2) * mo
            if mo > 0.:
                sd.append(stressdrop)
                mos.append(mo)
                print(mo, stressdrop)

        plt.scatter(mos, sd)
        plt.xlabel('Mo [N.m]')
        plt.ylabel('Stress drop [Pa]')
        plt.show()

        plt.hist(sd)
        plt.show()'''

        # calcul du point d'initiation
        deltawin = 3
        for i, mr in enumerate(mo_rate_list):
            idx = new_duration_dict[thresh][i]
            new_start = idx[0]
            new_end = idx[1]
            mr = mr[new_start:new_end]
            print(new_start, new_end)
            print(mr.shape)
            print(mr[0].shape)

            print('activated patches at time 0', np.sum(mr[0, :] != 0))
            print('activated patches at time 1', np.sum(mr[1, :] != 0))
            print('activated patches at time 2', np.sum(mr[2, :] != 0))

            def argmin_without_zero(x, axis=0):
                x_modified = np.where(x == 0.,np.inf, x)  # replace zeros with very high numbers to ignore them
                return np.argmin(x_modified, axis=axis)

            # patch actives au tps t --> mr[t, :] != 0
            ptime0 = mr[0, :] != 0
            ptimef = mr[-1, :] != 0
            print(np.argmax(mr[:deltawin], axis=1).shape)
            print(np.argmax(mr[:deltawin], axis=1))
            print(argmin_without_zero(mr[-deltawin:], axis=1))
            print(int(np.median(argmin_without_zero(mr[-deltawin:], axis=1))))


            ini_patch_idx = int(np.median(np.argmax(mr[:deltawin], axis=1)))
            term_patch_idx = int(np.median(np.argmax(mr[-deltawin:], axis=1)))
            x_ini_median, y_ini_median = se.ma.x_centr_lon[ini_patch_idx], se.ma.y_centr_lat[ini_patch_idx]
            x_term_median, y_term_median = se.ma.x_centr_lon[term_patch_idx], se.ma.y_centr_lat[term_patch_idx]



            plt.scatter(se.ma.x_centr_lon[ptime0], se.ma.y_centr_lat[ptime0], c=mr[:, ptime0].sum(axis=0), cmap='Blues')
            plt.scatter(x_ini_median, y_ini_median, marker='x', color='red')
            plt.show()
            plt.scatter(se.ma.x_centr_lon[ptimef], se.ma.y_centr_lat[ptimef], c=mr[:, ptimef].sum(axis=0), cmap='Blues')
            plt.scatter(se.ma.x_centr_lon[mr[-2, :] != 0], se.ma.y_centr_lat[mr[-2, :] != 0], c=mr[:, mr[-2, :] != 0].sum(axis=0), cmap='Oranges')
            plt.scatter(se.ma.x_centr_lon[mr[-3, :] != 0], se.ma.y_centr_lat[mr[-3, :] != 0], c=mr[:, mr[-3, :] != 0].sum(axis=0), cmap='Greens')
            plt.scatter(x_term_median, y_term_median, marker='x', color='red')
            plt.show()

    exit(0)

    TS = np.zeros((denoised_ts.shape[0], denoised_ts.shape[1], 3))
    TS[:, :, :2] = denoised_ts
    n_time_steps = TS.shape[0]

    slip_thresholds = [0.07, 0.3, 0.5, 0.7]
