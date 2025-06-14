import re


def ymd_decimal_year_lookup(from_decimal=False):
    """Returns a lookup table for (year, month, day) to decimal year, with the convention introduced by Nasa JPL."""
    ymd_decimal_lookup = dict()
    with open('../../DATA/common/date_utils/decyr.txt', 'r') as f:
        next(f)
        for line in f:
            line = re.sub(' +', ' ', line)
            splitted_line = line.split(' ')
            decimal, year, month, day = splitted_line[1], splitted_line[2], splitted_line[3], splitted_line[4]
            decimal, year, month, day = float(decimal), int(year), int(month), int(day)
            ymd_decimal_lookup[(year, month, day)] = decimal
    if not from_decimal:
        return ymd_decimal_lookup
    else:
        inv_lookuo = {v: k for k, v in ymd_decimal_lookup.items()}
        return inv_lookuo
