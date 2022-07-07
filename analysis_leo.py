import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_regression import KernelReg
import itertools
import datetime as dt
import matplotlib.dates as mdates
from sklearn import preprocessing

DATA_FOLDER = 'G:/My Drive/Podcast/PNB/data/formatted/'


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


if __name__ == '__main__':
    podcast_colours = {'Dropzilla': '#C5252C',
                       'Wasabicast': '#02AF7E',
                       'PressStartCast': '#117C91',
                       'NoJapao': '#FF3A11'}
    data_hand = pd.read_csv(DATA_FOLDER + 'data_handextracted.csv')
    with open(DATA_FOLDER + 'data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open(DATA_FOLDER + 'meta.pickle', 'rb') as f:
        meta = pickle.load(f)
    meta.ep_release_date = pd.to_datetime(meta.ep_release_date)
    data['TotalPlays'] = data['TotalPlays'].sort_values(by=['podcast', 'time'])
    podcasts = data['TotalPlays'].podcast.unique()

    rolls = []
    for podcast in podcasts:
        roll = data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast].plays.rolling(window=7).mean()
        rolls.append(roll)
    rolls = list(itertools.chain.from_iterable(rolls))
    data['TotalPlays']['plays_rolling_mean'] = rolls

    plays_smooth_golay = []
    for podcast in podcasts:
        plays = data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast].set_index('time').plays
        plays_smoothed_golay = savitzky_golay(plays, 365, 3)
        plays_smooth_golay.append(plays_smoothed_golay)
    data['TotalPlays']['plays_smooth_golay'] = list(itertools.chain.from_iterable(plays_smooth_golay))

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

    norm = []
    for podcast in podcasts:
        minmaxscaler = preprocessing.MinMaxScaler()
        plays = data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast].plays.values.reshape(-1, 1)
        normed = minmaxscaler.fit_transform(plays).reshape(1,-1)[0]
        norm.append(normed)
    data['TotalPlays']['plays_norm'] = list(itertools.chain.from_iterable(norm))

    norm = []
    for podcast in podcasts:
        minmaxscaler = preprocessing.MinMaxScaler()
        plays = data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast].plays_rolling_mean.values.reshape(-1, 1)
        normed = minmaxscaler.fit_transform(plays).reshape(1,-1)[0]
        norm.append(normed)
    data['TotalPlays']['plays_rolling_mean_norm'] = list(itertools.chain.from_iterable(norm))

    norm = []
    for podcast in podcasts:
        minmaxscaler = preprocessing.MinMaxScaler()
        plays = data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast].plays_smooth_golay.values.reshape(-1, 1)
        normed = minmaxscaler.fit_transform(plays).reshape(1,-1)[0]
        norm.append(normed)
    data['TotalPlays']['plays_smooth_golay_norm'] = list(itertools.chain.from_iterable(norm))

    plt.figure()
    for podcast in podcasts:
        data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast].set_index('time').plays_rolling_mean.plot(
            c=podcast_colours[podcast], alpha=0.3)
    # for podcast in podcasts:
    #     data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast].set_index('time').plays_smooth.plot(
    #         c=podcast_colours[podcast])
    for podcast in podcasts:
        data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast].set_index('time').plays_smooth_golay.plot(
            c=podcast_colours[podcast])
    # for podcast in podcasts:
    #     for release in meta.loc[meta.podcast == podcast].ep_release_date:
    #         plt.axvline(pd.Timestamp(release), color=podcast_colours[podcast], alpha=0.2)
    plt.axvspan(xmin=pd.Timestamp('2021-12-06'), xmax=pd.Timestamp('2021-12-11'), color='black', alpha=0.2)
    plt.text(pd.Timestamp('2021-12-06'), plt.gca().get_ylim()[1], 'First PNB Week',
             rotation=90, verticalalignment='top')
    plt.axvline(x=pd.Timestamp('2021-11-01'), color='black')
    plt.text(pd.Timestamp('2021-11-01'), plt.gca().get_ylim()[1], 'First Post of PNB on Instagram',
             rotation=90, verticalalignment='top')
    plt.legend(podcasts)
    plt.ylabel('Plays (smoothed and 7 days mov. avg.)')
    plt.xlabel('')
    plt.ylim(0)
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()

    plt.figure()
    for podcast in podcasts:
        ep_date = meta.loc[meta.podcast == podcast].sort_values(by='ep_release_date').ep_release_date
        y = data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast]\
            .merge(ep_date.rename('time'), on=['time'], how='right').set_index('time').plays_smooth_golay
        ep_duration = meta.loc[meta.podcast == podcast].sort_values(by='ep_release_date').set_index('ep_release_date')\
            .ep_duration_ms
        df = pd.concat([y, ep_duration], axis=1)
        df['ep_duration_ms'] = pd.to_numeric(df.ep_duration_ms)/(1000*60)
        sc = df.reset_index().plot(kind='scatter', x='index', y='plays_smooth_golay', s='ep_duration_ms', c=podcast_colours[podcast],
                                      ax=plt.gca(), alpha=0.5)
    leg_pod = plt.legend(podcasts)
    sizes = (pd.cut(df.ep_duration_ms, bins=4, retbins=True)[1]).round()
    for size in sizes:
        plt.scatter(x=[], y=[], s=(size), c='k', label=str(size))
    h, l = plt.gca().get_legend_handles_labels()
    plt.legend(h[:], l[:], labelspacing=0.5, title="Duration (min)", borderpad=0.5,
               frameon=True, framealpha=0.3, loc=4, numpoints=1)
    plt.gca().add_artist(leg_pod)
    plt.gca().xaxis.remove_overlapping_locs = False
    plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.ylabel('Plays (smoothed)')
    plt.xlabel('')
    plt.ylim(0)
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()

    plt.figure()
    # for podcast in podcasts:
    #     data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast].set_index('time').plays_rolling_mean_norm.plot(
    #         c=podcast_colours[podcast], alpha=0.3)
    # for podcast in podcasts:
    #     data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast].set_index('time').plays_smooth.plot(
    #         c=podcast_colours[podcast])
    for podcast in podcasts:
        data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast].set_index('time').plays_smooth_golay_norm.plot(
            c=podcast_colours[podcast])
    # for podcast in podcasts:
    #     for release in meta.loc[meta.podcast == podcast].ep_release_date:
    #         plt.axvline(pd.Timestamp(release), color=podcast_colours[podcast], alpha=0.2)
    plt.axvspan(xmin=pd.Timestamp('2021-12-06'), xmax=pd.Timestamp('2021-12-11'), color='black', alpha=0.2)
    plt.text(pd.Timestamp('2021-12-06'), plt.gca().get_ylim()[1], 'First PNB Week',
             rotation=90, verticalalignment='top')
    plt.axvline(x=pd.Timestamp('2021-11-01'), color='black')
    plt.text(pd.Timestamp('2021-11-01'), plt.gca().get_ylim()[1], 'First Post of PNB on Instagram',
             rotation=90, verticalalignment='top')
    plt.legend(podcasts)
    plt.ylabel('Normalized Plays (smoothed and 7 days mov. avg.)')
    plt.xlabel('')
    plt.ylim(0)
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()

    plt.figure()
    for podcast in podcasts:
        ep_date = meta.loc[meta.podcast == podcast].sort_values(by='ep_release_date').ep_release_date
        y = data['TotalPlays'].loc[data['TotalPlays'].podcast == podcast] \
            .merge(ep_date.rename('time'), on=['time'], how='right').set_index('time').plays_smooth_golay_norm
        ep_duration = meta.loc[meta.podcast == podcast].sort_values(by='ep_release_date').set_index('ep_release_date') \
            .ep_duration_ms
        df = pd.concat([y, ep_duration], axis=1)
        df['ep_duration_ms'] = pd.to_numeric(df.ep_duration_ms) / (1000 * 60)
        sc = df.reset_index().plot(kind='scatter', x='index', y='plays_smooth_golay_norm', s='ep_duration_ms',
                                   c=podcast_colours[podcast],
                                   ax=plt.gca(), alpha=0.5)
    leg_pod = plt.legend(podcasts)
    sizes = (pd.cut(df.ep_duration_ms, bins=4, retbins=True)[1]).round()
    for size in sizes:
        plt.scatter(x=[], y=[], s=(size), c='k', label=str(size))
    h, l = plt.gca().get_legend_handles_labels()
    plt.legend(h[:], l[:], labelspacing=0.5, title="Duration (min)", borderpad=0.5,
               frameon=True, framealpha=0.3, loc=4, numpoints=1)
    plt.gca().add_artist(leg_pod)
    plt.axvspan(xmin=pd.Timestamp('2021-12-06'), xmax=pd.Timestamp('2021-12-11'), color='black', alpha=0.2)
    plt.text(pd.Timestamp('2021-12-06'), plt.gca().get_ylim()[1], 'First PNB Week',
             rotation=90, verticalalignment='top')
    plt.axvline(x=pd.Timestamp('2021-11-01'), color='black')
    plt.text(pd.Timestamp('2021-11-01'), plt.gca().get_ylim()[1], 'First Post of PNB on Instagram',
             rotation=90, verticalalignment='top')
    plt.gca().xaxis.remove_overlapping_locs = False
    plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.ylabel('Normalized Plays (smoothed)')
    plt.xlabel('')
    plt.ylim(0)
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()


