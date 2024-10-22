import numpy as np

from utils.date_parsing import ymd_decimal_year_lookup


def tremor_catalogue(north=False):
    """Return tremors from PNSN + Ide (2012) tremor catalogue.
    The dates are in the JPL-style format."""
    tremors = []
    date_table = ymd_decimal_year_lookup()
    catalogue_names = ['tremor_catalogue', 'tremor_catalogue2']
    base_folder = '../../DATA/common/catalogues'
    for name in catalogue_names:
        with open(f'{base_folder}/{name}.txt') as f:
            next(f)
            for line in f:
                splitted_line = line.split(',')
                latitude = float(splitted_line[0].replace(' ', ''))
                longitude = float(splitted_line[1].replace(' ', ''))
                depth = float(splitted_line[2].replace(' ', ''))
                time = splitted_line[3][1:].split(' ')[0]
                year, month, day = time.split('-')
                year, month, day = float(year), float(month), float(day)
                decimal_date = date_table[(year, month, day)]
                tremors.append([latitude, longitude, depth, decimal_date])
    with open(f'{base_folder}/tremor_catalogue_ide.txt') as f:
        for line in f:
            splitted_line = line.split(',')
            latitude = float(splitted_line[2].replace(' ', ''))
            longitude = float(splitted_line[3].replace(' ', ''))
            depth = float(splitted_line[4].replace(' ', ''))
            year, month, day = splitted_line[0].split('-')
            year, month, day = float(year), float(month), float(day)
            decimal_date = date_table[(year, month, day)]
            tremors.append([latitude, longitude, depth, decimal_date])
    tremor_array = np.array(tremors)
    if north:
        valid_tremor_events = np.where(tremor_array[:, 0] >= 47.)[0]
        tremor_array = tremor_array[valid_tremor_events]
    return tremor_array
