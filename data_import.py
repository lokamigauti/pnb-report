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
import pickle
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os

DATA_FOLDER = 'G:/My Drive/Podcast/PNB/data/'
# DATA_FOLDER = 'C:/Users/Pedro/Desktop/Data analisys Podcast/data/'
OUTPUT_FOLDER = DATA_FOLDER + 'formatted/'
os.environ['SPOTIPY_CLIENT_ID'] = 'f001dec668494347ad43adb1accd9097'
os.environ['SPOTIPY_CLIENT_SECRET'] = 'INSERT_SECRET'
os.environ['SPOTIPY_REDIRECT_URI'] = 'http://localhost'

def debug_time(date_str):
    if date_str[-8:-6] != '24':
        return pd.to_datetime(date_str, format='%m/%d/%Y %H:%M:%S')

    date_str = date_str[:-8] + '00' + date_str[-6:]
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
    """
    Identify the source and content of the files in the data folder, import data and return a ndarray with the filepath,
    podcast name, content type, source, and data itself.
    :return: ndarray with data and metadata
    """
    filepaths = glob.glob(DATA_FOLDER + '*.csv')
    files = []
    for n, filepath in enumerate(filepaths):

        source = None
        content = None
        if re.search('TotalPlays_all-time', filepath):
            content = 'TotalPlays'
            source = 'Anchor'
            data = import_total_plays(filepath, source)
            podcast = re.search('(?<=\\\\)(.*)(?=_TotalPlays_all-time)', filepath).group(1)
        if re.search('TopEpisodes_all-time', filepath):
            content = 'TopEpisodes'
            source = 'Anchor'
            data = import_top_eps(filepath, source)
            podcast = re.search('(?<=\\\\)(.*)(?=_TopEpisodes_all-time)', filepath).group(1)
        if re.search('PlaysByDevice_all-time', filepath):
            content = 'PlaysByDevice'
            source = 'Anchor'
            data = import_plays_by_device(filepath, source)
            podcast = re.search('(?<=\\\\)(.*)(?=_PlaysByDevice_all-time)', filepath).group(1)
        if re.search('PlaysByApp_all-time', filepath):
            content = 'PlaysByApp'
            source = 'Anchor'
            data = import_plays_by_app(filepath, source)
            podcast = re.search('(?<=\\\\)(.*)(?=_PlaysByApp_all-time)', filepath).group(1)
        if re.search('GeoLocation_all-time', filepath):
            content = 'GeoLocation'
            source = 'Anchor'
            data = import_geolocation(filepath, source)
            podcast = re.search('(?<=\\\\)(.*)(?=_GeoLocation_all-time)', filepath).group(1)

        aggr = [filepath, podcast, content, source, data]
        files.append(aggr)

    return np.array(files, dtype=object)


def aggregate_data(files):
    """
    Aggregate data from identify_source_and_content() by content, creating the variable "podcast". Returns a dict with
    content in keys and pd.dataframes in values.
    :param files: ndarray from identify_source_and_content()
    :return: dict of pd.dataframes
    """
    contents = np.unique(files[:, 2])
    data = {}
    for content in contents:
        filter_arr = files[:, 2] == content
        content_file = files[filter_arr, :]
        for n_podcast in range(len(content_file)):
            content_file[n_podcast, 4]['podcast'] = content_file[n_podcast, 1]
        data[content] = pd.concat(content_file[:, 4])
    return data


def import_spotify_meta():
    scope = 'user-read-playback-position'
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyOAuth(scope=scope))
    spotify_meta = pd.DataFrame(
        columns=['ep_name', 'ep_release_date', 'ep_description', 'ep_duration_ms', 'ep_images', 'podcast'])
    metadata = pd.read_csv(DATA_FOLDER + 'spotify_meta.txt')
    for podcast in metadata['podcast']:
        podcast_meta = []
        podcast_id = metadata.loc[metadata['podcast'] == podcast].podcast_id.values[0]
        results = spotify.show_episodes(podcast_id)
        eps = results['items']
        while results['next']:
            results = spotify.next(results)
            eps.extend(results['items'])
        for ep in eps:
            ep_meta = [ep['name'], ep['release_date'], ep['description'], ep['duration_ms'], ep['images']]
            podcast_meta.append(ep_meta)
        podcast_meta = pd.DataFrame(podcast_meta,
                                    columns=['ep_name', 'ep_release_date', 'ep_description', 'ep_duration_ms',
                                             'ep_images'])
        podcast_meta['podcast'] = podcast
        spotify_meta = pd.concat([spotify_meta, podcast_meta], ignore_index=True)
    return spotify_meta


if __name__ == '__main__':
    data = aggregate_data(identify_source_and_content())
    meta = import_spotify_meta()

    with open(OUTPUT_FOLDER + 'data.pickle', 'wb') as f:
        pickle.dump(data, f)

    with open(OUTPUT_FOLDER + 'meta.pickle', 'wb') as f:
        pickle.dump(meta, f)

    # To load the data:
    # with open(OUTPUT_FOLDER + 'data.pickle', 'rb') as f:
    #     data = pickle.load(f)
