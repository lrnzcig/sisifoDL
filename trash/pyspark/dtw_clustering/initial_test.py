import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
# https://github.com/pierre-rouanet/dtw
#from dtw import accelerated_dtw
# https://github.com/paul-freeman/dtw
#from dtw import dtw
# https://pypi.org/project/dtw-python/
from dtw import dtw
import pickle
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from math import log, ceil
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import scale
from pandas.plotting import register_matplotlib_converters
from statistics import median


register_matplotlib_converters()

data = pd.read_csv("data/sales_agg_sample_4000.csv")


# 0. funciones auxiliares
def _trans_log(x):
    if x >= 0:
        return log(x + 1)
    else:
        return (-1) * log(1 - x)


def trans_log(x):
    return [_trans_log(el) for el in x]

# 1. calcular matriz de distancias

# 1.1. primero un ejemplo
"""
d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(np.array(data.iloc[0][[c for c in data.columns if c != "CUENTA"]]),
                                                        np.array(data.iloc[1][[c for c in data.columns if c != "CUENTA"]]),
                                                        dist='euclidean')
"""

# 1.2. con DTW solo se puede hacer bucle; buscar implementación matricial, buscar alternativa??
#      tarda media hora para 1000x1000 !!!
distance_matrix_path = "data/dtw_distance_matrix_4000x4000_log_sample_dtw_python.pickle"
if Path(distance_matrix_path).is_file():
    with open(distance_matrix_path, "rb") as f:
        distance_matrix = pickle.load(f)
else:
    size = data.shape[0]
    yearmonth_columns = [c for c in data.columns if c != "CUENTA"]
    distance_matrix = np.empty((size, size))
    for coord_x in range(0, size):
        print("x=" + str(coord_x))
        for coord_y in range(0, size):
            if coord_x == coord_y:
                distance_matrix[coord_x, coord_y] = 0
            elif coord_x > coord_y:
                distance_matrix[coord_x, coord_y] = distance_matrix[coord_y, coord_x]
            else:
                # https://github.com/pierre-rouanet/dtw
                """
                d, _, _, _ = accelerated_dtw(np.array(trans_log(data.iloc[coord_x][yearmonth_columns])),
                                             np.array(trans_log(data.iloc[coord_y][yearmonth_columns])),
                                             dist='euclidean')
                                             """
                # https://github.com/paul-freeman/dtw
                """
                d, _, _, _ = dtw(np.array(trans_log(data.iloc[coord_x][yearmonth_columns])),
                                 np.array(trans_log(data.iloc[coord_y][yearmonth_columns])))
                                 """
                # https://pypi.org/project/dtw-python/
                d = dtw(np.array(trans_log(data.iloc[coord_x][yearmonth_columns])),
                        np.array(trans_log(data.iloc[coord_y][yearmonth_columns]))).distance

                distance_matrix[coord_x, coord_y] = d

    with open(distance_matrix_path, "wb") as f:
        pickle.dump(distance_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)


# 2. clustering aglomerativo a lo loco
n_clusters = 8
acm = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete', affinity='precomputed')
clustering = acm.fit(distance_matrix)
from collections import Counter
counts = Counter(clustering.labels_)
print(counts) # probablemente es espectral; también comprobar qué linkage es el adecuado (average vs complete vs single)

df2plot = data.drop('CUENTA', axis=1).transpose()
df2plot.index = pd.DatetimeIndex(pd.to_datetime(df2plot.index, format='%Y%m'))

fig, ax = plt.subplots(figsize=(15, 10), nrows=ceil(n_clusters/2), ncols=2)
fig.tight_layout()
big_clusters = []
big_threshold = 250
for cluster in range(0, n_clusters):
    for col in df2plot.columns[clustering.labels_ == cluster]:
        sns.lineplot(data=df2plot.loc[:, col], ax=ax[cluster // 2][cluster % 2])
    #sns.lineplot(data=df2plot.loc[:, df2plot.columns[clustering.labels_ == cluster]],
    #             ax=ax[cluster // 2][cluster % 2], dashes=False)
    #ax[cluster // 2][cluster % 2].legend_.remove()
    ax[cluster // 2][cluster % 2].set_title("cluster " + str(cluster))
    if counts[cluster] > big_threshold:
        big_clusters += [cluster]

fig, ax = plt.subplots(figsize=(15, 10), nrows=ceil(n_clusters/2), ncols=2)
fig.tight_layout()
q2plot = 5
for cluster in range(0, n_clusters):
    waves = df2plot.loc[:, df2plot.columns[clustering.labels_ == cluster]]
    sns.lineplot(data=waves.apply(median, axis=1),
                 ax=ax[cluster // 2][cluster % 2])
    low = waves.apply(np.percentile, q=q2plot, axis=1)
    high = waves.apply(np.percentile, q=(100-q2plot), axis=1)
    ax[cluster // 2][cluster % 2].fill_between(df2plot.index, low, high, alpha=.2)
    ax[cluster // 2][cluster % 2].set_title("cluster " + str(cluster))


def cluster_report(df2plot, cluster, ax, ax_index,
                   q2plot=5):
    print("cluster " + str(cluster))
    waves = df2plot.loc[:, df2plot.columns[clustering.labels_ == cluster]]
    means = np.mean(waves)
    d = seasonal_decompose(pd.DataFrame(scale(waves), index=waves.index))

    print("count: " + str(counts[cluster]))
    print("mean:  " + str(np.mean(means)))
    print("std:   " + str(np.std(means)))
    print("max:   " + str(np.max(means)))
    print("min:   " + str(np.min(means)))

    d.seasonal.apply(median, axis=1).plot(ax=ax[0][ax_index])
    low = d.seasonal.apply(np.percentile, q=q2plot, axis=1)
    high = d.seasonal.apply(np.percentile, q=(100-q2plot), axis=1)
    ax[0][ax_index].fill_between(df2plot.index, low, high, alpha=.2)
    ax[0][ax_index].set_title("estac. c" + str(cluster))
    ax[0][ax_index].set_xticklabels([])

    d.trend.apply(median, axis=1).plot(ax=ax[1][ax_index])
    low = d.trend.apply(np.percentile, q=q2plot, axis=1)
    high = d.trend.apply(np.percentile, q=(100-q2plot), axis=1)
    ax[1][ax_index].fill_between(df2plot.index, low, high, alpha=.2)
    ax[1][ax_index].set_title("tend. c " + str(cluster))


fig, ax = plt.subplots(figsize=(15, 10), nrows=2, ncols=len(big_clusters))
fig.tight_layout()
ax_counter = 0
for cluster in big_clusters:
    cluster_report(df2plot, cluster, ax, ax_counter,
                   q2plot=25)
    ax_counter += 1


fig, ax = plt.subplots(figsize=(15, 10), nrows=2, ncols=n_clusters - len(big_clusters))
fig.tight_layout()
ax_counter = 0
for cluster in [c for c in range(0, n_clusters) if c not in big_clusters]:
    cluster_report(df2plot, cluster, ax, ax_counter,
                   q2plot=25)
    ax_counter += 1


plt.show(block=True)
