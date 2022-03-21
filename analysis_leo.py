import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_regression import KernelReg
import itertools
import datetime as dt

DATA_FOLDER = 'G:/My Drive/Podcast/PNB/data/formatted/'

if __name__ == '__main__':
    with open(DATA_FOLDER + 'data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open(DATA_FOLDER + 'meta.pickle', 'rb') as f:
        meta = pickle.load(f)
    meta.ep_release_date = pd.to_datetime(meta.ep_release_date)
    podcasts = meta.podcast.unique()
    data['TotalPlays']['plays_rolling_mean'] = data['TotalPlays'].groupby('podcast').plays.rolling(
        window=7).mean().values

    podcast_colours = dict(Dropzilla='#C5252C', Wasabicast='#02AF7E')



    kr = []
    for podcast in podcasts:
        kr.append(KernelReg(data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast].plays,
                            data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast].time.map(dt.datetime.toordinal),
                            'c'))
    plays_smooth = []
    for n, podcast in enumerate(podcasts):
        plays_smoothed, std = kr[n].fit(data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast].time.map(dt.datetime.toordinal))
        plays_smooth.append(plays_smoothed)
    data['TotalPlays']['plays_smooth'] = list(itertools.chain.from_iterable(plays_smooth))

    plt.figure()
    for podcast in podcasts:
        data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast].set_index('time').plays_rolling_mean.plot(
            c=podcast_colours[podcast])
    for podcast in podcasts:
        data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast].set_index('time').plays_smooth.plot(
            c=podcast_colours[podcast])
    # for podcast in podcasts:
    #     for release in meta.loc[meta.podcast == podcast].ep_release_date:
    #         plt.axvline(pd.Timestamp(release), color=podcast_colours[podcast], alpha=0.2)
    plt.axvspan(xmin=pd.Timestamp('2021-12-06'), xmax=pd.Timestamp('2021-12-11'), color='black', alpha=0.2)
    plt.legend(podcasts)
    plt.ylabel('Plays (7 days mov. avg.)')
    plt.xlabel('')
    plt.ylim(0)
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()
