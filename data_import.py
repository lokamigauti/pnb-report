"""
Import, format and tidy data.

This script imports the data stored in the "data" folder, and save a .csv (or .nc, TBD.) with the formatted dataset.
The data sources can vary, therefore, the script needs to identify the source (not explicitly provided) and adapt to
them. Some meta-data (e.g. the podcast name) is in the file name itself, so the script needs to identify it as well.
"""

import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
import glob
import re

DATA_FOLDER = 'G:/My Drive/Podcast/PNB/data/'
#DATA_FOLDER = 'C:/Users/Pedro/Desktop/Data analisys Podcast/data/'
OUTPUT_FOLDER = DATA_FOLDER + 'formatted/'


def debug_time(date_str):
    if date_str[9:11] != '24':
        return pd.to_datetime(date_str, format='%m/%d/%Y %H:%M:%S')

    date_str = date_str[0:9] + '00' + date_str[11:]
    return pd.to_datetime(date_str, format='%m/%d/%Y %H:%M:%S') + dt.timedelta(days=1)


def import_total_plays(path, source):
    """
    Import a "total plays" file.
    :param path: str, path to file
    :param source: str, source of the file, e.g. 'Anchor'.
    :return: pandas dataframe with columns 'time' and 'plays', with '1D' interval.
    """

    sources_list = ['test', 'Anchor']

    assert type(path) == str, 'path must be a string: %r' % path
    assert type(source) == str, 'source must be a string: %r' % source
    assert source in sources_list, 'Source %r not implemented. Make sure the spelling is correct.' % source

    if source == 'test':
        data = None

    if source == 'Anchor':
        # by default, %H is 00-23, however we do not need the time info as it is daily.
        anchor_date_parser = lambda x: datetime.strptime(x, '%m/%d/%Y 24:00:00')
        data = pd.read_csv(path, names=['time', 'plays'], header=0, parse_dates=['time'],
                           date_parser=anchor_date_parser)

    return data


def import_top_eps(path, source):
    """
    Import a "top episodes" file.
    :param path: str, path to file
    :param source: str, source of the file, e.g. 'Anchor'.
    :return: pandas dataframe with columns 'time' and 'plays', with '1D' interval.
    """

    sources_list = ['test', 'Anchor']

    assert type(path) == str, 'path must be a string: %r' % path
    assert type(source) == str, 'source must be a string: %r' % source
    assert source in sources_list, 'Source %r not implemented. Make sure the spelling is correct.' % source

    if source == 'test':
        data = None

    if source == 'Anchor':
        data = pd.read_csv(path, names=['title', 'plays', 'time'], header=0)
        data.time = data.time.apply(debug_time)

    return data


def import_plays_by_device(path, source):
    """
    Import a "plays by devices" file.
    :param path: str, path to file
    :param source: str, source of the file, e.g. 'Anchor'.
    :return: pandas dataframe with columns 'time' and 'plays', with '1D' interval.
    """

    sources_list = ['test', 'Anchor']

    assert type(path) == str, 'path must be a string: %r' % path
    assert type(source) == str, 'source must be a string: %r' % source
    assert source in sources_list, 'Source %r not implemented. Make sure the spelling is correct.' % source

    if source == 'test':
        data = None

    if source == 'Anchor':
        data = pd.read_csv(path, names=['device', 'plays_perc'], header=0)

    return data


def import_plays_by_app(path, source):
    """
    Import a "plays by app" file.
    :param path: str, path to file
    :param source: str, source of the file, e.g. 'Anchor'.
    :return: pandas dataframe with columns 'time' and 'plays', with '1D' interval.
    """

    sources_list = ['test', 'Anchor']

    assert type(path) == str, 'path must be a string: %r' % path
    assert type(source) == str, 'source must be a string: %r' % source
    assert source in sources_list, 'Source %r not implemented. Make sure the spelling is correct.' % source

    if source == 'test':
        data = None

    if source == 'Anchor':
        data = pd.read_csv(path, names=['app', 'plays_perc'], header=0)

    return data


def import_geolocation(path, source):
    """
    Import a "geolocation" file.
    :param path: str, path to file
    :param source: str, source of the file, e.g. 'Anchor'.
    :return: pandas dataframe with columns 'time' and 'plays', with '1D' interval.
    """

    sources_list = ['test', 'Anchor']

    assert type(path) == str, 'path must be a string: %r' % path
    assert type(source) == str, 'source must be a string: %r' % source
    assert source in sources_list, 'Source %r not implemented. Make sure the spelling is correct.' % source

    if source == 'test':
        data = None

    if source == 'Anchor':
        data = pd.read_csv(path, names=['location', 'plays_perc'], header=0)

    return data


def identify_source_and_content():
    filepaths = glob.glob(DATA_FOLDER + '*.csv')
    files = []
    for n, filepath in enumerate(filepaths):

        source = None
        content = None
        if re.search('TotalPlays_all-time', filepath):
            source = 'Anchor'
            content = import_total_plays(filepath, source)
            podcast = re.search('(?<=\\\\)(.*)(?=_TotalPlays_all-time)', filepath).group(1)
        if re.search('TopEpisodes_all-time', filepath):
            source = 'Anchor'
            content = import_top_eps(filepath, source)
            podcast = re.search('(?<=\\\\)(.*)(?=_TopEpisodes_all-time)', filepath).group(1)
        if re.search('PlaysByDevice_all-time', filepath):
            source = 'Anchor'
            content = import_plays_by_device(filepath, source)
            podcast = re.search('(?<=\\\\)(.*)(?=_PlaysByDevice_all-time)', filepath).group(1)
        if re.search('PlaysByApp_all-time', filepath):
            source = 'Anchor'
            content = import_plays_by_app(filepath, source)
            podcast = re.search('(?<=\\\\)(.*)(?=_PlaysByApp_all-time)', filepath).group(1)
        if re.search('GeoLocation_all-time', filepath):
            source = 'Anchor'
            content = import_geolocation(filepath, source)
            podcast = re.search('(?<=\\\\)(.*)(?=_GeoLocation_all-time)', filepath).group(1)

        data = [filepath, podcast, content, source]
        files.append(data)

    return files


if __name__ == '__main__':
    files = identify_source_and_content()
