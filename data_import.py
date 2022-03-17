"""
Import, format and tidy data.

This script imports the data stored in the "data" folder, and save a .csv (or .nc, TBD.) with the formatted dataset.
The data sources can vary, therefore, the script needs to identify the source (not explicitly provided) and adapt to
them. Some meta-data (e.g. the podcast name) is in the file name itself, so the script needs to identify it as well.
"""

import numpy as np
import pandas as pd
from types import * # types is safe for import *
from datetime import datetime

DATA_FOLDER = 'G:/My Drive/Podcast/PNB/data/'
OUTPUT_FOLDER = DATA_FOLDER + 'formatted/'


def import_total_plays(path, source):
    """
    Import a "total plays" file.
    :param path: str, path to file
    :param source: str, source of the file, e.g. 'Anchor'.
    :return: pandas dataframe with columns 'time' and 'plays', with '1D' interval.
    """

    sources_list = ['test', 'Anchor']

    assert type(path) is str, 'path must be a string: %r' % path
    assert type(source) is str, 'source must be a string: %r' % source
    assert source in sources_list, 'Source %r not implemented. Make sure the spelling is correct.' % source

    if source is 'test':
        total_plays = None

    if source is 'Anchor':
        # by default, %H is 00-23, however we do not need the time info as it is daily.
        anchor_date_parser = lambda x: datetime.strptime(x, '%m/%d/%Y 24:00:00')
        total_plays = pd.read_csv(path, names=['time', 'plays'], header=0, parse_dates=['time'],
                                  date_parser=anchor_date_parser)

    return total_plays


if __name__ == '__main__':
    total_plays = import_total_plays(DATA_FOLDER + 'Dropzilla_TotalPlays_all-time.csv', 'Anchor')