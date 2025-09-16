import matplotlib.pyplot as plt
import numpy as np

from config_files.plotting_style import set_matplotlib_style
from sse_extraction.SlowSlipEventExtractor import SlowSlipEventExtractor

set_matplotlib_style()

if __name__ == '__main__':
    se = SlowSlipEventExtractor()
    time_array = se.ddh.corrected_time
    print(se.ma.slip_rates.shape)
    '''[plt.plot(time_array, sr) for sr in se.ma.slip_rates.T]
    plt.axhline(y=0.07)
    plt.xlabel('Time [years]')
    plt.ylabel('Slip rate [mm/day]')
    plt.show()'''

    '''[plt.plot(time_array, sr) for sr in se.ma.slip_rates.T]
    plt.axhline(y=0.07)
    plt.xlabel('Time [years]')
    plt.ylabel('Slip rate [mm/day]')
    plt.ylim([0, .1])
    plt.show()'''

    slip_rates = se.ma.slip_rates.ravel()
    print(slip_rates.shape)
    print('Median:', np.median(slip_rates), 'mm/day ->', 365*np.median(slip_rates), 'mm/year')
    print('ratio threshold/median', 0.07/np.median(slip_rates))
    plt.hist(slip_rates, bins=10000)
    # [plt.hist(sr, bins=50) for sr in se.ma.slip_rates.T]
    plt.xlabel('Slip rate [mm/day]')
    plt.ylabel('Count')
    plt.axvline(x=0.07, color='k', label='Threshold of 0.07 mmd$^{-1}$')
    plt.axvline(x=np.median(slip_rates), color='r', label='Median')
    plt.xlim([-0.002, .2])
    # plt.xscale('log')
    #plt.yscale('log')
    plt.legend()
    plt.savefig(f'figures/histogram_thresh.pdf', bbox_inches='tight')
    # plt.show()